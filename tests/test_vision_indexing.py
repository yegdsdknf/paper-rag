import json
import tempfile
import unittest
from pathlib import Path


class VisionIndexingTest(unittest.TestCase):
    def test_select_vision_pages_combines_forced_noisy_and_dense_reasons(self):
        from paper_rag.config import RagSettings
        from paper_rag.indexing.vision import PageVisionSignal, select_vision_pages

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
                "vision_visual_density_threshold": 0.35,
                "vision_max_pages_per_doc": 2,
                "vision_force_pages": {"paper.pdf": [5]},
            }
        )
        signals = [
            PageVisionSignal(source_file="paper.pdf", page=1, visual_density=0.4, quality_flags=[]),
            PageVisionSignal(source_file="paper.pdf", page=2, visual_density=0.1, quality_flags=["unicode_escape_noise"]),
            PageVisionSignal(source_file="paper.pdf", page=5, visual_density=0.0, quality_flags=[]),
        ]

        selected = select_vision_pages(signals, settings)

        self.assertEqual([item.page for item in selected], [5, 2])
        self.assertEqual(selected[0].trigger_reasons, ["forced_page"])
        self.assertEqual(selected[1].trigger_reasons, ["unicode_escape_noise"])

    def test_select_vision_pages_respects_forced_only_policy(self):
        from paper_rag.config import RagSettings
        from paper_rag.indexing.vision import PageVisionSignal, select_vision_pages

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
                "vision_trigger_policy": "forced_only",
                "vision_visual_density_threshold": 0.35,
                "vision_force_pages": {"paper.pdf": [5]},
            }
        )
        signals = [
            PageVisionSignal(source_file="paper.pdf", page=1, visual_density=0.8, quality_flags=[]),
            PageVisionSignal(source_file="paper.pdf", page=2, visual_density=0.1, quality_flags=["unicode_escape_noise"]),
            PageVisionSignal(source_file="paper.pdf", page=5, visual_density=0.0, quality_flags=[]),
        ]

        selected = select_vision_pages(signals, settings)

        self.assertEqual([item.page for item in selected], [5])
        self.assertEqual(selected[0].trigger_reasons, ["forced_page"])

    def test_summary_cache_round_trip_records_cache_hit(self):
        from paper_rag.indexing.vision import VisionCache

        with tempfile.TemporaryDirectory() as tmp:
            cache = VisionCache(tmp)
            cached = cache.load("hash-a", 5, "qwen2.5vl:3b", "v1")
            self.assertIsNone(cached)

            cache.save("hash-a", 5, "qwen2.5vl:3b", "v1", "summary", "image.png")
            cached = cache.load("hash-a", 5, "qwen2.5vl:3b", "v1")

            self.assertEqual(cached["summary"], "summary")
            self.assertEqual(cached["image_path"], "image.png")
            self.assertTrue(Path(cached["cache_file"]).exists())

    def test_build_vision_summary_document_has_required_metadata(self):
        from paper_rag.indexing.vision import build_vision_summary_document

        doc = build_vision_summary_document(
            source_path="papers/deepseekr1.pdf",
            source_file_hash="md5-deepseek",
            page=51,
            summary="页面类型：figure\n主要内容：展示 R1 训练流程。",
            image_path="data/vision_cache/images/deepseekr1_p0051.png",
            model="qwen2.5vl:3b",
            prompt_version="v1",
            cache_hit=False,
            trigger_reasons=["forced_page", "figure_dense_page"],
            quality_flags=["vision_generated"],
        )

        self.assertIn("R1", doc.page_content)
        self.assertEqual(doc.metadata["block_type"], "vision_summary")
        self.assertEqual(doc.metadata["chunk_strategy"], "vision_summary")
        self.assertEqual(doc.metadata["vision_model"], "qwen2.5vl:3b")
        self.assertEqual(doc.metadata["vision_prompt_version"], "v1")
        self.assertEqual(doc.metadata["image_path"], "data/vision_cache/images/deepseekr1_p0051.png")
        self.assertFalse(doc.metadata["vision_cache_hit"])
        self.assertEqual(doc.metadata["vision_trigger_reason"], ["forced_page", "figure_dense_page"])
        self.assertEqual(doc.metadata["quality_flags"], ["vision_generated"])
        self.assertEqual(doc.metadata["source_file"], "deepseekr1.pdf")
        self.assertEqual(doc.metadata["page"], 51)
        self.assertTrue(doc.metadata["chunk_id"].startswith("doc_md5-deepseek:vision_summary:v1:"))

    def test_collect_vision_summaries_reuses_cache_before_calling_model(self):
        from paper_rag.config import RagSettings
        from paper_rag.indexing.vision import collect_vision_summary_docs

        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "vision_cache"
            images_dir = cache_dir / "images"
            images_dir.mkdir(parents=True)
            image_path = images_dir / "paper_p0003.png"
            image_path.write_bytes(b"fake")
            cache_file = cache_dir / "summaries" / "hash-a_p0003_qwen2_5vl_3b_v1.json"
            cache_file.parent.mkdir(parents=True)
            cache_file.write_text(
                json.dumps({"summary": "cached summary", "image_path": str(image_path)}, ensure_ascii=False),
                encoding="utf-8",
            )
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
                    "vision_cache_dir": str(cache_dir),
                    "vision_force_pages": {"paper.pdf": [3]},
                }
            )

            docs, stats = collect_vision_summary_docs(
                pdf_paths=["papers/paper.pdf"],
                settings=settings,
                file_hashes={"papers/paper.pdf": "hash-a"},
                quality_reports=[],
                render_page=lambda *_args, **_kwargs: str(image_path),
                density_detector=lambda *_args, **_kwargs: 0.0,
                summarizer=lambda *_args, **_kwargs: self.fail("summarizer should not be called"),
            )

            self.assertEqual(len(docs), 1)
            self.assertEqual(docs[0].page_content, "cached summary")
            self.assertTrue(docs[0].metadata["vision_cache_hit"])
            self.assertEqual(stats["cache_hits"], 1)
            self.assertEqual(stats["generated"], 0)
            self.assertEqual(stats["quality_flags"], {})

    def test_collect_vision_summaries_records_quality_flag_stats(self):
        from paper_rag.config import RagSettings
        from paper_rag.indexing.vision import collect_vision_summary_docs

        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "vision_cache"
            image_path = cache_dir / "images" / "paper_p0004.png"
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
                    "vision_cache_dir": str(cache_dir),
                }
            )

            docs, stats = collect_vision_summary_docs(
                pdf_paths=["papers/paper.pdf"],
                settings=settings,
                file_hashes={"papers/paper.pdf": "hash-a"},
                quality_reports=[
                    {
                        "source": "paper.pdf",
                        "page": 4,
                        "reason": "unicode_escape_noise",
                        "quality_flags": ["unicode_escape_noise"],
                    }
                ],
                render_page=lambda *_args, **_kwargs: str(image_path),
                density_detector=lambda *_args, **_kwargs: 0.0,
                summarizer=lambda *_args, **_kwargs: "generated summary",
            )

            self.assertEqual(len(docs), 1)
            self.assertEqual(stats["quality_flags"], {"unicode_escape_noise": 1})


if __name__ == "__main__":
    unittest.main()
