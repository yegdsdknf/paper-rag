import unittest


class RagSettingsTest(unittest.TestCase):
    def test_from_mapping_requires_core_fields(self):
        from paper_rag.config import ConfigError, RagSettings

        with self.assertRaises(ConfigError) as ctx:
            RagSettings.from_mapping({"llm_model": "qwen2.5:3b"})

        self.assertIn("persist_directory", str(ctx.exception))
        self.assertIn("embedding_model", str(ctx.exception))

    def test_from_mapping_applies_stable_defaults(self):
        from paper_rag.config import RagSettings

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
            }
        )

        self.assertEqual(settings.llm_num_ctx, 4096)
        self.assertEqual(settings.llm_num_predict, 1024)
        self.assertEqual(settings.query_log_path, "logs/query_runs.jsonl")
        self.assertEqual(settings.chunk_strategy, "recursive_character")
        self.assertEqual(settings.chunk_schema_version, "v1")
        self.assertEqual(settings.index_manifest_filename, "index_manifest.json")
        self.assertEqual(settings.skip_pages, {})
        self.assertEqual(settings.as_dict()["llm_model"], "qwen2.5:3b")
        self.assertEqual(settings.as_dict()["chunk_strategy"], "recursive_character")

    def test_get_setting_supports_mapping_and_typed_settings(self):
        from paper_rag.config import RagSettings, get_setting

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
            }
        )

        self.assertEqual(get_setting({"k": 3}, "k", 6), 3)
        self.assertEqual(get_setting(settings, "k", 3), 6)
        self.assertEqual(get_setting(settings, "missing", "fallback"), "fallback")


if __name__ == "__main__":
    unittest.main()
