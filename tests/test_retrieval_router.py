import unittest
from unittest.mock import Mock

from langchain_core.documents import Document

from retrieval_router import RetrievalRouter, is_comparison_question, is_evidence_question, is_overview_question


class FakeRetriever:
    def __init__(self, docs):
        self.docs = docs

    def invoke(self, query):
        return self.docs


class FakeHybrid:
    def __init__(self, docs_by_query=None):
        self.docs_by_query = docs_by_query or {}
        self.queries = []

    def get_retriever(self, query):
        self.queries.append(query)
        return FakeRetriever(self.docs_by_query.get(query, []))


class FakeVectorStore:
    def get(self, include=None):
        docs = []
        metadatas = []
        for source in ["bert.pdf", "gpt3.pdf", "t5.pdf", "attention is all you need.pdf"]:
            for page in range(4):
                content = f"{source} page {page}"
                if source == "attention is all you need.pdf" and page == 0:
                    content = "Attention Is All You Need. We propose a new simple network architecture, the Transformer."
                docs.append(content)
                metadatas.append({"source": source, "page": page})
        return {"documents": docs, "metadatas": metadatas}


class FakeHybridWithStore(FakeHybrid):
    vector_store = FakeVectorStore()


class FakeRemoteEvidenceVectorStore:
    def get(self, include=None):
        return {
            "documents": [
                "Language Models are Few-Shot Learners. GPT-3 has 175 billion parameters.",
                "General architecture notes about the transformer layers.",
                "Few-shot demonstrations are specified purely via text interaction with the model.",
                "Unrelated appendix content.",
            ],
            "metadatas": [
                {"source": "gpt3.pdf", "page": 0},
                {"source": "gpt3.pdf", "page": 7},
                {"source": "gpt3.pdf", "page": 12},
                {"source": "bert.pdf", "page": 9},
            ],
        }


class FakeHybridWithRemoteEvidence(FakeHybrid):
    vector_store = FakeRemoteEvidenceVectorStore()


class RetrievalRouterTest(unittest.TestCase):
    def test_question_classifiers_keep_current_route_signals(self):
        self.assertTrue(is_comparison_question("BERT 和 GPT-3 的区别是什么？"))
        self.assertTrue(is_overview_question("什么是 Transformer？"))
        self.assertTrue(is_evidence_question("GPT-3 的证据在哪一页？"))
        self.assertFalse(is_comparison_question("说明论文训练流程"))

    def test_overview_question_uses_mixed_route_and_rerank(self):
        docs = [
            Document(page_content="first", metadata={"source": "a.pdf", "page": 1}),
            Document(page_content="second", metadata={"source": "b.pdf", "page": 2}),
        ]
        rerank = Mock(return_value=list(reversed(docs)))
        router = RetrievalRouter(
            settings={"enable_query_expansion": False, "enable_rerank": True, "rerank_top_k": 2},
            apply_rerank_fn=rerank,
            embedding_device_fn=lambda: "cpu",
        )

        routed_docs, strategy = router.route(FakeHybrid({"什么是 BERT？": docs}), "什么是 BERT？")

        self.assertEqual(strategy, "mixed")
        self.assertEqual([doc.metadata["source"] for doc in routed_docs], ["b.pdf", "a.pdf"])
        rerank.assert_called_once()

    def test_router_accepts_typed_settings(self):
        from paper_rag.config import RagSettings

        docs = [
            Document(page_content="first", metadata={"source": "a.pdf", "page": 1}),
            Document(page_content="second", metadata={"source": "b.pdf", "page": 2}),
        ]
        settings = RagSettings.from_mapping(
            {
                "persist_directory": "./chroma_db",
                "embedding_model": "BAAI/bge-m3",
                "llm_model": "qwen2.5:3b",
                "temperature": 0.1,
                "k": 6,
                "chunk_size": 500,
                "chunk_overlap": 100,
                "separators": ["\n\n", "\n"],
                "enable_query_expansion": False,
                "enable_rerank": True,
                "rerank_top_k": 1,
            }
        )
        rerank = Mock(return_value=docs[:1])
        router = RetrievalRouter(settings=settings, apply_rerank_fn=rerank)

        routed_docs, strategy = router.route(FakeHybrid({"什么是 BERT？": docs}), "什么是 BERT？")

        self.assertEqual(strategy, "mixed")
        self.assertEqual(len(routed_docs), 1)
        rerank.assert_called_once()

    def test_non_overview_question_uses_hyde_route(self):
        hyde_docs = [Document(page_content="hyde")]
        router = RetrievalRouter(
            settings={},
            hyde_retrieve_fn=Mock(return_value=hyde_docs),
        )

        docs, strategy = router.route(FakeHybrid(), "说明论文训练流程")

        self.assertEqual(strategy, "hyde")
        self.assertEqual(docs, hyde_docs)

    def test_evidence_question_filters_to_mentioned_source_and_skips_rerank(self):
        candidates = [
            Document(page_content="generic transformer", metadata={"source": "bert.pdf", "page": 3}),
            Document(page_content="gpt architecture", metadata={"source": "gpt3.pdf", "page": 7}),
        ]
        rerank = Mock(return_value=candidates)
        router = RetrievalRouter(
            settings={"enable_query_expansion": True, "rerank_top_k": 5},
            multi_query_retrieve_fn=Mock(return_value=(candidates, ["variant"])),
            apply_rerank_fn=rerank,
        )

        docs, strategy = router.route(FakeHybridWithStore(), "GPT-3 使用 Transformer 结构的证据在哪一页？")

        self.assertEqual(strategy, "mixed_multi_query")
        self.assertEqual(
            [(doc.metadata["source"], doc.metadata["page"]) for doc in docs],
            [("gpt3.pdf", 0), ("gpt3.pdf", 7)],
        )
        rerank.assert_not_called()

    def test_evidence_question_adds_matching_remote_evidence_page(self):
        docs = [Document(page_content="gpt3 abstract", metadata={"source": "gpt3.pdf", "page": 0})]
        router = RetrievalRouter(
            settings={"enable_query_expansion": False, "enable_rerank": False, "rerank_top_k": 5},
        )

        routed_docs, strategy = router.route(
            FakeHybridWithRemoteEvidence({"GPT-3 使用 Transformer 结构的证据在哪一页？": docs}),
            "GPT-3 使用 Transformer 结构的证据在哪一页？",
        )

        self.assertEqual(strategy, "mixed")
        self.assertIn(
            ("gpt3.pdf", 7),
            [(doc.metadata["source"], doc.metadata["page"]) for doc in routed_docs[:5]],
        )

    def test_generic_transformer_overview_pins_origin_paper_front_page(self):
        docs = [
            Document(page_content="deepseek appendix", metadata={"source": "deepseekr1.pdf", "page": 78}),
            Document(page_content="transformer conclusion", metadata={"source": "attention is all you need.pdf", "page": 10}),
        ]
        router = RetrievalRouter(
            settings={"enable_query_expansion": False, "enable_rerank": False, "rerank_top_k": 5},
        )

        routed_docs, strategy = router.route(
            FakeHybridWithStore({"Transformer 是基于什么机制构建的？": docs}),
            "Transformer 是基于什么机制构建的？",
        )

        self.assertEqual(strategy, "mixed")
        self.assertEqual(
            [(doc.metadata["source"], doc.metadata["page"]) for doc in routed_docs[:2]],
            [("attention is all you need.pdf", 0), ("attention is all you need.pdf", 10)],
        )

    def test_generic_transformer_detail_does_not_crowd_out_later_relevant_page(self):
        docs = [
            Document(page_content="sinusoidal equations", metadata={"source": "attention is all you need.pdf", "page": 8}),
            Document(page_content="positional encoding section", metadata={"source": "attention is all you need.pdf", "page": 5}),
        ]
        router = RetrievalRouter(
            settings={"enable_query_expansion": False, "enable_rerank": False, "rerank_top_k": 5},
        )

        routed_docs, _ = router.route(
            FakeHybridWithStore({"Transformer 为什么要加入 positional encodings？": docs}),
            "Transformer 为什么要加入 positional encodings？",
        )

        self.assertIn(
            ("attention is all you need.pdf", 5),
            [(doc.metadata["source"], doc.metadata["page"]) for doc in routed_docs[:5]],
        )

    def test_known_source_detail_question_uses_mixed_route_with_front_page_anchor(self):
        docs = [
            Document(page_content="late corpus appendix", metadata={"source": "t5.pdf", "page": 38}),
            Document(page_content="late corpus table", metadata={"source": "t5.pdf", "page": 40}),
        ]
        hyde = Mock(return_value=[Document(page_content="hyde")])
        router = RetrievalRouter(
            settings={"enable_query_expansion": False, "enable_rerank": False, "rerank_top_k": 5},
            hyde_retrieve_fn=hyde,
        )

        routed_docs, strategy = router.route(
            FakeHybridWithStore({"T5 使用的语料规模和名字是什么？": docs}),
            "T5 使用的语料规模和名字是什么？",
        )

        self.assertEqual(strategy, "mixed")
        self.assertEqual((routed_docs[0].metadata["source"], routed_docs[0].metadata["page"]), ("t5.pdf", 0))
        hyde.assert_not_called()

    def test_known_source_questions_include_early_neighbor_pages_for_multi_evidence(self):
        docs = [Document(page_content="bert abstract", metadata={"source": "bert.pdf", "page": 0})]
        router = RetrievalRouter(
            settings={"enable_query_expansion": False, "enable_rerank": False, "rerank_top_k": 5},
        )

        routed_docs, _ = router.route(
            FakeHybridWithStore({"BERT 为什么叫双向 Transformer？": docs}),
            "BERT 为什么叫双向 Transformer？",
        )

        self.assertEqual(
            [(doc.metadata["source"], doc.metadata["page"]) for doc in routed_docs[:4]],
            [("bert.pdf", 0), ("bert.pdf", 1), ("bert.pdf", 2), ("bert.pdf", 3)],
        )

    def test_known_source_question_adds_matching_remote_evidence_page(self):
        docs = [Document(page_content="gpt3 abstract", metadata={"source": "gpt3.pdf", "page": 0})]
        router = RetrievalRouter(
            settings={"enable_query_expansion": False, "enable_rerank": False, "rerank_top_k": 5},
        )

        routed_docs, _ = router.route(
            FakeHybridWithRemoteEvidence({"GPT-3 论文中的 few-shot 示例是通过什么形式提供给模型的？": docs}),
            "GPT-3 论文中的 few-shot 示例是通过什么形式提供给模型的？",
        )

        self.assertIn(
            ("gpt3.pdf", 12),
            [(doc.metadata["source"], doc.metadata["page"]) for doc in routed_docs[:5]],
        )

    def test_summary_question_uses_source_front_page_terms_for_remote_evidence(self):
        docs = [Document(page_content="gpt3 abstract", metadata={"source": "gpt3.pdf", "page": 0})]
        router = RetrievalRouter(
            settings={"enable_query_expansion": False, "enable_rerank": False, "rerank_top_k": 5},
        )

        routed_docs, _ = router.route(
            FakeHybridWithRemoteEvidence({"GPT-3 的核心特点是什么？": docs}),
            "GPT-3 的核心特点是什么？",
        )

        self.assertIn(
            ("gpt3.pdf", 12),
            [(doc.metadata["source"], doc.metadata["page"]) for doc in routed_docs[:5]],
        )

    def test_remote_evidence_keeps_multiple_matching_pages_within_top_five(self):
        docs = [Document(page_content="gpt3 abstract", metadata={"source": "gpt3.pdf", "page": 0})]

        class MultiPageVectorStore:
            def get(self, include=None):
                return {
                    "documents": [
                        "Language Models are Few-Shot Learners.",
                        "In-context learning abilities.",
                        "Few-shot completion examples.",
                        "Few-shot demonstrations specified purely via text interaction.",
                    ],
                    "metadatas": [
                        {"source": "gpt3.pdf", "page": 0},
                        {"source": "gpt3.pdf", "page": 4},
                        {"source": "gpt3.pdf", "page": 11},
                        {"source": "gpt3.pdf", "page": 12},
                    ],
                }

        class MultiPageHybrid(FakeHybrid):
            vector_store = MultiPageVectorStore()

        router = RetrievalRouter(
            settings={"enable_query_expansion": False, "enable_rerank": False, "rerank_top_k": 5},
        )

        routed_docs, _ = router.route(
            MultiPageHybrid({"GPT-3 论文中的 few-shot 示例是通过什么形式提供给模型的？": docs}),
            "GPT-3 论文中的 few-shot 示例是通过什么形式提供给模型的？",
        )

        self.assertIn(
            ("gpt3.pdf", 12),
            [(doc.metadata["source"], doc.metadata["page"]) for doc in routed_docs[:5]],
        )


if __name__ == "__main__":
    unittest.main()
