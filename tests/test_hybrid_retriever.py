import unittest

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


if __name__ == "__main__":
    unittest.main()
