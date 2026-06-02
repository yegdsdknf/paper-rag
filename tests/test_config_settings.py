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
        self.assertTrue(settings.section_heading_detection)
        self.assertEqual(settings.section_max_chars, 900)
        self.assertEqual(settings.semantic_min_chars, 180)
        self.assertEqual(settings.semantic_max_chars, 900)
        self.assertEqual(settings.skip_pages, {})
        self.assertFalse(settings.enable_vision_analysis)
        self.assertEqual(settings.vision_model, "qwen2.5vl:3b")
        self.assertEqual(settings.vision_prompt_version, "v1")
        self.assertEqual(settings.vision_cache_dir, "./data/vision_cache")
        self.assertFalse(settings.vision_force_refresh)
        self.assertEqual(settings.vision_trigger_policy, "noisy_or_figure_page")
        self.assertEqual(settings.vision_visual_density_threshold, 0.35)
        self.assertEqual(settings.vision_max_pages_per_doc, 20)
        self.assertEqual(settings.vision_force_pages, {})
        self.assertEqual(settings.as_dict()["llm_model"], "qwen2.5:3b")
        self.assertEqual(settings.as_dict()["chunk_strategy"], "recursive_character")
        self.assertEqual(settings.as_dict()["references_policy"], "keep_with_metadata")
        self.assertEqual(settings.as_dict()["vision_model"], "qwen2.5vl:3b")

    def test_from_mapping_normalizes_vision_force_pages(self):
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
                "enable_vision_analysis": True,
                "vision_force_pages": {"deepseekr1.pdf": ["51", 52], "bad": "not-a-list"},
            }
        )

        self.assertTrue(settings.enable_vision_analysis)
        self.assertEqual(settings.vision_force_pages, {"deepseekr1.pdf": [51, 52]})

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
