import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from hybrid_retriever import HybridRetriever


class EmptyVectorStore:
    def __init__(self):
        self.vector_retriever = object()

    def get(self, include=None):
        if include == []:
            return {"ids": []}
        return {"documents": [], "metadatas": []}

    def as_retriever(self, search_type=None, search_kwargs=None):
        return self.vector_retriever


class HybridRetrieverTest(unittest.TestCase):
    def test_empty_bm25_documents_fall_back_to_vector_retriever(self):
        vector_store = EmptyVectorStore()
        hybrid = HybridRetriever(vector_store=vector_store, embedding_model=None)

        self.assertIsNone(hybrid.build_bm25_retriever())
        self.assertIs(hybrid.get_retriever("anything"), vector_store.vector_retriever)

    def test_weight_log_includes_query_context_and_reason(self):
        vector_store = EmptyVectorStore()
        hybrid = HybridRetriever(vector_store=vector_store, embedding_model=None)
        hybrid.build_bm25_retriever = lambda: object()

        with patch("hybrid_retriever.EnsembleRetriever", return_value=object()):
            output = StringIO()
            with redirect_stdout(output):
                hybrid.get_retriever("GPT-3 175B parameters", log_context="[query original]")

        text = output.getvalue()
        self.assertIn("[query original]", text)
        self.assertIn("原因=", text)
        self.assertIn("精确信号", text)


if __name__ == "__main__":
    unittest.main()
