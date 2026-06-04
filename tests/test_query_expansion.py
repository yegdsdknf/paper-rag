import unittest
from unittest.mock import patch

from langchain_core.documents import Document

import rag_pipeline
from query_expansion import expand_query, filter_query_variants


class FakeExpansionLLM:
    def invoke(self, prompt):
        return type(
            "Response",
            (),
            {
                "content": """
1. self-attention mechanisms in transformer papers
2. 自注意力机制在 Transformer 论文中的证据
3. should be ignored
"""
            },
        )()


class NoisyExpansionLLM:
    def invoke(self, prompt):
        return type(
            "Response",
            (),
            {
                "content": """
查询1:
1. GPT-3 Transformer architecture evidence page
2. same model and architecture as GPT-2 transformer layers
"""
            },
        )()


class PrefixedExpansionLLM:
    def invoke(self, prompt):
        return type(
            "Response",
            (),
            {
                "content": """
query1: GPT-3 transformer architecture evidence page
query2: same model and architecture as GPT-2 transformer layers
"""
            },
        )()


class FakeHybrid:
    def __init__(self):
        self.queries = []

    def get_retriever(self, query):
        self.queries.append(query)

        class Retriever:
            def invoke(self, query):
                return {
                    "compare attention query": [
                        Document(page_content="original", metadata={"source": "a.pdf", "page": 1}),
                        Document(page_content="shared", metadata={"source": "b.pdf", "page": 2}),
                    ],
                    "self-attention mechanisms in transformer papers": [
                        Document(page_content="shared variant", metadata={"source": "b.pdf", "page": 2}),
                        Document(page_content="variant", metadata={"source": "c.pdf", "page": 3}),
                    ],
                    "自注意力机制在 Transformer 论文中的证据": [
                        Document(page_content="extra", metadata={"source": "d.pdf", "page": 4}),
                        Document(page_content="overflow", metadata={"source": "e.pdf", "page": 5}),
                    ],
                }[query]

        return Retriever()


class FakeEmbeddings:
    def embed_query(self, text):
        vectors = {
            "compare attention query": [1.0, 0.0],
            "compare attention query duplicate": [0.999, 0.001],
            "self-attention mechanisms in transformer papers": [0.8, 0.2],
            "unrelated cooking recipe": [0.0, 1.0],
        }
        return vectors[text]


class FakeWeightDecider:
    embeddings = FakeEmbeddings()


class FakeHybridWithEmbeddings(FakeHybrid):
    weight_decider = FakeWeightDecider()

    def get_retriever(self, query):
        self.queries.append(query)

        class Retriever:
            def invoke(self, query):
                return {
                    "compare attention query": [
                        Document(page_content="original", metadata={"source": "a.pdf", "page": 1}),
                    ],
                    "self-attention mechanisms in transformer papers": [
                        Document(page_content="variant", metadata={"source": "c.pdf", "page": 3}),
                    ],
                }[query]

        return Retriever()


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


class QueryExpansionTest(unittest.TestCase):
    def test_expand_query_keeps_original_out_of_variants_and_limits_count(self):
        variants = expand_query("attention query", FakeExpansionLLM(), n_variants=2)

        self.assertEqual(
            variants,
            [
                "self-attention mechanisms in transformer papers",
                "自注意力机制在 Transformer 论文中的证据",
            ],
        )

    def test_expand_query_filters_label_only_variants(self):
        variants = expand_query("GPT-3 使用 Transformer 结构的证据在哪一页？", NoisyExpansionLLM(), n_variants=2)

        self.assertEqual(
            variants,
            [
                "GPT-3 Transformer architecture evidence page",
                "same model and architecture as GPT-2 transformer layers",
            ],
        )

    def test_expand_query_strips_label_prefixes_from_variants(self):
        variants = expand_query("GPT-3 使用 Transformer 结构的证据在哪一页？", PrefixedExpansionLLM(), n_variants=2)

        self.assertEqual(
            variants,
            [
                "GPT-3 transformer architecture evidence page",
                "same model and architecture as GPT-2 transformer layers",
            ],
        )

    def test_filter_query_variants_rejects_too_near_and_too_far_embeddings(self):
        result = filter_query_variants(
            "compare attention query",
            [
                "compare attention query duplicate",
                "self-attention mechanisms in transformer papers",
                "unrelated cooking recipe",
            ],
            embed_fn=FakeEmbeddings().embed_query,
            enabled=True,
            min_similarity=0.3,
            max_similarity=0.98,
        )

        self.assertEqual(result.variants, ["self-attention mechanisms in transformer papers"])
        self.assertEqual(
            [(item["variant"], item["reason"]) for item in result.rejections],
            [
                ("compare attention query duplicate", "too_similar"),
                ("unrelated cooking recipe", "too_distant"),
            ],
        )

    def test_mixed_route_filters_query_variants_before_retrieval(self):
        class FilteredExpansionLLM:
            def invoke(self, prompt):
                return type(
                    "Response",
                    (),
                    {
                        "content": """
1. compare attention query duplicate
2. self-attention mechanisms in transformer papers
3. unrelated cooking recipe
"""
                    },
                )()

        fake_hybrid = FakeHybridWithEmbeddings()
        with (
            patch.dict(
                rag_pipeline.config,
                {
                    "enable_query_expansion": True,
                    "query_expansion_variants": 3,
                    "query_expansion_max_multiplier": 3,
                    "enable_query_expansion_similarity_filter": True,
                    "query_expansion_min_similarity": 0.3,
                    "query_expansion_max_similarity": 0.98,
                    "enable_rerank": False,
                },
            ),
            patch.object(rag_pipeline, "_get_llm", return_value=FilteredExpansionLLM()),
        ):
            docs, variants = rag_pipeline._retrieve_multi_query(fake_hybrid, "compare attention query")

        self.assertEqual(
            fake_hybrid.queries,
            ["compare attention query", "self-attention mechanisms in transformer papers"],
        )
        self.assertEqual(variants, ["self-attention mechanisms in transformer papers"])
        self.assertEqual([doc.metadata["source"] for doc in docs], ["a.pdf", "c.pdf"])

    def test_mixed_route_uses_multi_query_before_single_rerank(self):
        captured = {}

        def fake_rerank(query, docs, **kwargs):
            captured["query"] = query
            captured["docs"] = docs
            return docs

        fake_hybrid = FakeHybrid()
        with (
            patch.dict(
                rag_pipeline.config,
                {
                    "enable_query_expansion": True,
                    "query_expansion_variants": 2,
                    "query_expansion_max_multiplier": 3,
                    "enable_rerank": True,
                },
            ),
            patch.object(rag_pipeline, "_get_llm", return_value=FakeExpansionLLM()),
            patch.object(rag_pipeline, "apply_rerank", side_effect=fake_rerank) as rerank,
        ):
            docs, strategy = rag_pipeline._route_retrieve(fake_hybrid, "compare attention query")

        self.assertEqual(strategy, "mixed_multi_query")
        self.assertEqual(
            fake_hybrid.queries,
            [
                "compare attention query",
                "self-attention mechanisms in transformer papers",
                "自注意力机制在 Transformer 论文中的证据",
            ],
        )
        self.assertEqual(
            [doc.metadata["source"] for doc in captured["docs"]],
            ["a.pdf", "b.pdf", "c.pdf", "d.pdf", "e.pdf"],
        )
        self.assertLessEqual(len(captured["docs"]), 2 * 3)
        self.assertEqual(docs, captured["docs"])
        rerank.assert_called_once()

    def test_evidence_question_uses_multi_query_route(self):
        with (
            patch.dict(
                rag_pipeline.config,
                {
                    "enable_query_expansion": True,
                    "query_expansion_variants": 2,
                    "query_expansion_max_multiplier": 3,
                    "enable_rerank": True,
                },
            ),
            patch.object(rag_pipeline, "_retrieve_multi_query", return_value=([Document(page_content="evidence")], ["variant"])),
            patch.object(rag_pipeline, "_retrieve_with_hyde") as hyde,
            patch.object(rag_pipeline, "apply_rerank", side_effect=lambda query, docs, **kwargs: docs),
        ):
            docs, strategy = rag_pipeline._route_retrieve(FakeHybrid(), "GPT-3 使用 Transformer 结构的证据在哪一页？")

        self.assertEqual(strategy, "mixed_multi_query")
        self.assertEqual(docs[0].page_content, "evidence")
        hyde.assert_not_called()

    def test_evidence_question_filters_to_mentioned_source_and_skips_rerank(self):
        candidates = [
            Document(page_content="generic transformer", metadata={"source": "t5.pdf", "page": 3}),
            Document(page_content="gpt architecture", metadata={"source": "gpt3.pdf", "page": 7}),
            Document(page_content="attention", metadata={"source": "attention is all you need.pdf", "page": 1}),
        ]

        with (
            patch.dict(rag_pipeline.config, {"enable_rerank": True, "rerank_top_k": 5}),
            patch.object(rag_pipeline, "_retrieve_multi_query", return_value=(candidates, ["variant"])),
            patch.object(rag_pipeline, "apply_rerank") as rerank,
        ):
            docs, strategy = rag_pipeline._route_retrieve(FakeHybridWithStore(), "GPT-3 使用 Transformer 结构的证据在哪一页？")

        self.assertEqual(strategy, "mixed_multi_query")
        self.assertEqual(
            [(doc.metadata["source"], doc.metadata["page"]) for doc in docs],
            [("gpt3.pdf", 0), ("gpt3.pdf", 7)],
        )
        rerank.assert_not_called()

    def test_compare_question_pins_front_page_anchors_after_rerank(self):
        reranked = [
            Document(page_content="late bert", metadata={"source": "bert.pdf", "page": 13}),
            Document(page_content="late gpt", metadata={"source": "gpt3.pdf", "page": 19}),
        ]

        with (
            patch.dict(rag_pipeline.config, {"enable_rerank": True, "rerank_top_k": 5}),
            patch.object(rag_pipeline, "_retrieve_multi_query", return_value=(reranked, ["variant"])),
            patch.object(rag_pipeline, "apply_rerank", return_value=reranked),
        ):
            docs, strategy = rag_pipeline._route_retrieve(FakeHybridWithStore(), "BERT 和 GPT-3 的架构差异是什么？")

        self.assertEqual(strategy, "mixed_multi_query")
        self.assertEqual(
            [(doc.metadata["source"], doc.metadata["page"]) for doc in docs],
            [
                ("bert.pdf", 0),
                ("gpt3.pdf", 0),
                ("bert.pdf", 1),
                ("bert.pdf", 2),
                ("bert.pdf", 3),
            ],
        )

    def test_compare_commonality_adds_transformer_origin_anchor(self):
        reranked = [Document(page_content="gpt result", metadata={"source": "gpt3.pdf", "page": 13})]

        with (
            patch.dict(rag_pipeline.config, {"enable_rerank": True, "rerank_top_k": 5}),
            patch.object(rag_pipeline, "_retrieve_multi_query", return_value=(reranked, ["variant"])),
            patch.object(rag_pipeline, "apply_rerank", return_value=reranked),
        ):
            docs, _ = rag_pipeline._route_retrieve(FakeHybridWithStore(), "T5 和 GPT-3 的共同点是什么？")

        self.assertEqual(
            [(doc.metadata["source"], doc.metadata["page"]) for doc in docs[:3]],
            [
                ("t5.pdf", 0),
                ("gpt3.pdf", 0),
                ("attention is all you need.pdf", 0),
            ],
        )


if __name__ == "__main__":
    unittest.main()
