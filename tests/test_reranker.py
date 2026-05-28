import unittest
from unittest.mock import patch

from langchain_core.documents import Document

from reranker import Reranker, apply_rerank


class FakeScorer:
    def predict(self, pairs):
        scores = []
        for _, text in pairs:
            if "best" in text:
                scores.append(0.9)
            elif "middle" in text:
                scores.append(0.5)
            else:
                scores.append(0.1)
        return scores


class RerankerTest(unittest.TestCase):
    def test_rerank_orders_documents_by_model_score_and_keeps_metadata(self):
        docs = [
            Document(page_content="weak evidence", metadata={"source": "a.pdf", "page": 1}),
            Document(page_content="best evidence", metadata={"source": "b.pdf", "page": 2}),
            Document(page_content="middle evidence", metadata={"source": "c.pdf", "page": 3}),
        ]

        reranker = Reranker(model=FakeScorer())
        ranked = reranker.rerank("query", docs, top_k=2)

        self.assertEqual([doc.metadata["source"] for doc in ranked], ["b.pdf", "c.pdf"])
        self.assertEqual(ranked[0].metadata["rerank_score"], 0.9)

    def test_apply_rerank_returns_original_docs_when_disabled(self):
        docs = [Document(page_content="best evidence")]

        self.assertIs(apply_rerank("query", docs, enabled=False), docs)

    def test_apply_rerank_falls_back_to_original_docs_when_model_unavailable(self):
        docs = [Document(page_content="best evidence")]

        with patch("reranker.get_reranker", side_effect=RuntimeError("offline")):
            self.assertIs(apply_rerank("query", docs, enabled=True), docs)


if __name__ == "__main__":
    unittest.main()
