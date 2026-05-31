import json
import tempfile
import unittest
from pathlib import Path

from paper_rag.config import RagSettings


def make_settings(**overrides):
    data = {
        "persist_directory": "./chroma_db",
        "embedding_model": "BAAI/bge-m3",
        "llm_model": "qwen2.5:3b",
        "temperature": 0.1,
        "k": 6,
        "chunk_size": 500,
        "chunk_overlap": 100,
        "separators": ["\n\n", "\n"],
    }
    data.update(overrides)
    return RagSettings.from_mapping(data)


class IndexManifestTest(unittest.TestCase):
    def test_build_index_manifest_records_versioned_build_inputs(self):
        from paper_rag.indexing.manifest import build_index_manifest

        settings = make_settings()
        manifest = build_index_manifest(
            settings=settings,
            source_files=["papers/bert.pdf"],
            file_hashes={"papers/bert.pdf": "md5-bert"},
            chunk_count=12,
            embedding_device="cuda",
            created_at="2026-05-31T00:00:00+00:00",
        )

        self.assertTrue(manifest["index_version"].startswith("idx_"))
        self.assertEqual(manifest["embedding_model"], "BAAI/bge-m3")
        self.assertEqual(manifest["chunk_strategy"], "recursive_character")
        self.assertEqual(manifest["chunk_schema_version"], "v1")
        self.assertEqual(manifest["chunk_count"], 12)
        self.assertEqual(manifest["embedding_device"], "cuda")
        self.assertEqual(manifest["source_files"][0]["filename"], "bert.pdf")
        self.assertEqual(manifest["source_files"][0]["file_hash"], "md5-bert")

    def test_save_load_and_resolve_index_manifest(self):
        from paper_rag.indexing.manifest import (
            load_index_manifest,
            resolve_index_version,
            save_index_manifest,
        )

        with tempfile.TemporaryDirectory() as tmp:
            settings = make_settings(persist_directory=tmp)
            manifest = {
                "index_version": "idx_test123",
                "embedding_model": "BAAI/bge-m3",
            }

            path = save_index_manifest(manifest, settings)
            self.assertEqual(path, Path(tmp) / "index_manifest.json")
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), manifest)
            self.assertEqual(load_index_manifest(settings), manifest)
            self.assertEqual(resolve_index_version(settings), "idx_test123")

    def test_resolve_index_version_falls_back_to_config_fingerprint(self):
        from paper_rag.indexing.manifest import build_index_version, resolve_index_version

        settings = make_settings(persist_directory="./missing-index-dir")

        self.assertEqual(resolve_index_version(settings), build_index_version(settings))


if __name__ == "__main__":
    unittest.main()
