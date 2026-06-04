import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class FakeVectorStore:
    def __init__(self, persist_directory, documents=None, metadatas=None):
        self._persist_directory = persist_directory
        self._collection_name = "langchain"
        self.documents = documents or ["BERT pretraining", "GPT autoregressive"]
        self.metadatas = metadatas or [{"source": "bert.pdf"}, {"source": "gpt3.pdf"}]
        self.full_get_calls = 0

    def get(self, include=None):
        if include == []:
            return {"ids": [str(i) for i in range(len(self.documents))]}
        if include == ["documents", "metadatas"]:
            self.full_get_calls += 1
            return {"documents": self.documents, "metadatas": self.metadatas}
        return {}


class HybridRetrieverBm25CacheTest(unittest.TestCase):
    def test_uses_persisted_bm25_cache_without_full_chroma_scan(self):
        from hybrid_retriever import HybridRetriever
        from paper_rag.retrieval.bm25_cache import build_bm25_cache_metadata, save_bm25_cache

        with tempfile.TemporaryDirectory() as tmp:
            manifest = {
                "chunk_count": 2,
                "chunk_schema_version": "v4",
                "source_files": [{"filename": "bert.pdf", "file_hash": "hash"}],
            }
            metadata = build_bm25_cache_metadata(
                persist_directory=tmp,
                collection_name="langchain",
                chunk_schema_version="v4",
                doc_count=2,
                top_k=2,
                manifest=manifest,
            )
            cached_retriever = {"cached": True}
            save_bm25_cache(Path(tmp) / "bm25_cache", cached_retriever, metadata)

            vector_store = FakeVectorStore(tmp)
            hybrid = HybridRetriever(vector_store=vector_store, top_k=2, chunk_schema_version="v4")

            with patch("hybrid_retriever.load_index_manifest", return_value=manifest):
                retriever = hybrid.build_bm25_retriever()

        self.assertEqual(retriever, cached_retriever)
        self.assertEqual(vector_store.full_get_calls, 0)

    def test_rebuilds_and_saves_cache_when_metadata_mismatches(self):
        from hybrid_retriever import HybridRetriever
        from langchain_community.retrievers import BM25Retriever

        with tempfile.TemporaryDirectory() as tmp:
            old_manifest = {
                "chunk_count": 1,
                "chunk_schema_version": "v4",
                "source_files": [{"filename": "old.pdf", "file_hash": "old"}],
            }
            new_manifest = {
                "chunk_count": 2,
                "chunk_schema_version": "v4",
                "source_files": [{"filename": "new.pdf", "file_hash": "new"}],
            }
            vector_store = FakeVectorStore(tmp)
            hybrid = HybridRetriever(vector_store=vector_store, top_k=2, chunk_schema_version="v4")

            with patch("hybrid_retriever.load_index_manifest", return_value=old_manifest):
                hybrid.build_bm25_retriever()

            hybrid.bm25_retriever = None
            hybrid._bm25_doc_count = 0

            with patch("hybrid_retriever.load_index_manifest", return_value=new_manifest):
                retriever = hybrid.build_bm25_retriever()

        self.assertIsInstance(retriever, BM25Retriever)
        self.assertEqual(vector_store.full_get_calls, 2)


if __name__ == "__main__":
    unittest.main()
