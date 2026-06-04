import tempfile
import unittest
from pathlib import Path


class Bm25CacheMetadataTest(unittest.TestCase):
    def test_build_cache_metadata_uses_manifest_fingerprint(self):
        from paper_rag.retrieval.bm25_cache import build_bm25_cache_metadata

        manifest = {
            "chunk_count": 2,
            "chunk_schema_version": "v4",
            "source_files": [
                {"filename": "b.pdf", "file_hash": "hash-b"},
                {"filename": "a.pdf", "file_hash": "hash-a"},
            ],
        }

        metadata = build_bm25_cache_metadata(
            persist_directory="./db",
            collection_name="langchain",
            chunk_schema_version="v4",
            doc_count=2,
            top_k=6,
            manifest=manifest,
        )

        self.assertEqual(metadata["collection_name"], "langchain")
        self.assertEqual(metadata["chunk_count"], 2)
        self.assertEqual(metadata["chunk_schema_version"], "v4")
        self.assertEqual(metadata["top_k"], 6)
        self.assertTrue(metadata["source_fingerprint"])
        self.assertEqual(metadata["cache_format_version"], 1)

    def test_cache_round_trip_and_metadata_validation(self):
        from paper_rag.retrieval.bm25_cache import load_bm25_cache, save_bm25_cache

        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "bm25_cache"
            metadata = {
                "collection_name": "langchain",
                "persist_directory": tmp,
                "chunk_schema_version": "v4",
                "chunk_count": 2,
                "source_fingerprint": "fp",
                "top_k": 6,
                "created_at": "2026-06-04T00:00:00+00:00",
                "cache_format_version": 1,
                "bm25_class": "FakeRetriever",
            }
            retriever = {"fake": "retriever"}

            save_bm25_cache(cache_dir, retriever, metadata)

            hit = load_bm25_cache(cache_dir, metadata)
            miss = load_bm25_cache(cache_dir, {**metadata, "top_k": 8})

        self.assertEqual(hit, retriever)
        self.assertIsNone(miss)

    def test_corrupt_pickle_is_ignored(self):
        from paper_rag.retrieval.bm25_cache import load_bm25_cache, save_bm25_cache

        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "bm25_cache"
            metadata = {
                "collection_name": "langchain",
                "persist_directory": tmp,
                "chunk_schema_version": "v4",
                "chunk_count": 2,
                "source_fingerprint": "fp",
                "top_k": 6,
                "created_at": "2026-06-04T00:00:00+00:00",
                "cache_format_version": 1,
                "bm25_class": "FakeRetriever",
            }
            save_bm25_cache(cache_dir, {"fake": "retriever"}, metadata)
            (cache_dir / "bm25_retriever.pkl").write_bytes(b"not-a-pickle")

            loaded = load_bm25_cache(cache_dir, metadata)

        self.assertIsNone(loaded)


if __name__ == "__main__":
    unittest.main()
