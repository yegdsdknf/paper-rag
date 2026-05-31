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
        for source in ["bert.pdf", "gpt3.pdf", "attention is all you need.pdf"]:
            for page in range(4):
                docs.append(f"{source} page {page}")
                metadatas.append({"source": source, "page": page})
        return {"documents": docs, "metadatas": metadatas}


class FakeHybridWithStore(FakeHybrid):
    vector_store = FakeVectorStore()


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


if __name__ == "__main__":
    unittest.main()
