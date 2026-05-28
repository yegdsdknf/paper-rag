import unittest
from unittest.mock import patch

from langchain_core.documents import Document

import rag_pipeline


class FakeHybrid:
    def get_retriever(self, query):
        class Retriever:
            def invoke(self, query):
                return [
                    Document(page_content="first", metadata={"source": "a.pdf", "page": 1}),
                    Document(page_content="second", metadata={"source": "b.pdf", "page": 2}),
                ]

        return Retriever()


class RagRerankIntegrationTest(unittest.TestCase):
    def test_mixed_route_applies_rerank(self):
        def fake_apply(query, docs, **kwargs):
            return list(reversed(docs))

        with patch.object(rag_pipeline, "apply_rerank", side_effect=fake_apply) as rerank:
            docs, strategy = rag_pipeline._route_retrieve(FakeHybrid(), "BERT 和 ViT 有什么区别")

        self.assertEqual(strategy, "mixed")
        self.assertEqual([doc.metadata["source"] for doc in docs], ["b.pdf", "a.pdf"])
        rerank.assert_called_once()

    def test_hyde_route_does_not_apply_rerank_yet(self):
        with (
            patch.object(rag_pipeline, "_retrieve_with_hyde", return_value=[Document(page_content="hyde")]),
            patch.object(rag_pipeline, "apply_rerank") as rerank,
        ):
            docs, strategy = rag_pipeline._route_retrieve(FakeHybrid(), "说明论文采用的训练流程")

        self.assertEqual(strategy, "hyde")
        self.assertEqual(docs[0].page_content, "hyde")
        rerank.assert_not_called()


if __name__ == "__main__":
    unittest.main()
