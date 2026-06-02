import unittest
from contextlib import redirect_stderr
from io import StringIO

from langchain_core.documents import Document

from paper_rag.config import RagSettings


def make_settings(**overrides):
    data = {
        "persist_directory": "./chroma_db",
        "collection_name": "langchain",
        "embedding_model": "BAAI/bge-m3",
        "llm_model": "qwen2.5:3b",
        "temperature": 0.1,
        "k": 6,
        "chunk_size": 500,
        "chunk_overlap": 100,
        "separators": ["\n\n", "\n"],
        "chunk_strategy": "recursive_character",
        "chunk_schema_version": "v1",
    }
    data.update(overrides)
    return RagSettings.from_mapping(data)


class BuildExperimentTest(unittest.TestCase):
    def test_parse_build_args_supports_experiment_rebuild_and_chunk_overrides(self):
        from build_knowledge import parse_build_args

        args = parse_build_args(
            [
                "--experiment",
                "section-aware",
                "--chunk-strategy",
                "section_aware",
                "--chunk-schema-version",
                "v2",
                "--batch-size",
                "16",
                "--rebuild",
            ]
        )

        self.assertEqual(args.experiment, "section-aware")
        self.assertEqual(args.chunk_strategy, "section_aware")
        self.assertEqual(args.chunk_schema_version, "v2")
        self.assertEqual(args.batch_size, 16)
        self.assertTrue(args.rebuild)

    def test_resolve_build_plan_isolates_experiment_storage_and_dedup(self):
        from build_knowledge import parse_build_args, resolve_build_plan

        args = parse_build_args(
            [
                "--experiment",
                "section-aware",
                "--chunk-strategy",
                "section_aware",
                "--chunk-schema-version",
                "v2",
            ]
        )
        plan = resolve_build_plan(make_settings(), args)

        self.assertEqual(plan.settings.chunk_strategy, "section_aware")
        self.assertEqual(plan.settings.chunk_schema_version, "v2")
        self.assertEqual(plan.settings.persist_directory, "./chroma_db_experiments/section_aware")
        self.assertEqual(plan.settings.collection_name, "langchain_section_aware")
        self.assertEqual(plan.dedup_record_path, "./data/md5_records_section_aware.json")
        self.assertFalse(plan.rebuild)
        self.assertEqual(plan.batch_size, 64)

    def test_resolve_build_plan_applies_batch_size_override(self):
        from build_knowledge import parse_build_args, resolve_build_plan

        args = parse_build_args(["--experiment", "small-batch", "--batch-size", "16"])
        plan = resolve_build_plan(make_settings(), args)

        self.assertEqual(plan.batch_size, 16)

    def test_resolve_build_plan_keeps_default_storage_without_experiment(self):
        from build_knowledge import parse_build_args, resolve_build_plan

        plan = resolve_build_plan(make_settings(), parse_build_args([]))

        self.assertEqual(plan.settings.persist_directory, "./chroma_db")
        self.assertEqual(plan.settings.collection_name, "langchain")
        self.assertEqual(plan.dedup_record_path, "./data/md5_records.json")

    def test_rebuild_requires_experiment_name(self):
        from build_knowledge import parse_build_args

        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parse_build_args(["--rebuild"])

    def test_get_new_papers_does_not_mark_dedup_until_success(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from build_knowledge import get_new_papers
        from utils.dedup_manager import DedupManager

        with tempfile.TemporaryDirectory() as tmp:
            papers_dir = Path(tmp) / "papers"
            papers_dir.mkdir()
            paper = papers_dir / "demo.pdf"
            paper.write_bytes(b"pdf")
            record_path = Path(tmp) / "records.json"

            with patch("build_knowledge.PDF_DIR", str(papers_dir)):
                new_papers = get_new_papers(str(record_path))

            self.assertEqual([Path(path).name for path in new_papers], ["demo.pdf"])
            self.assertEqual(DedupManager(str(record_path)).get_all_records(), {})

    def test_mark_papers_indexed_records_dedup_after_success(self):
        import tempfile
        from pathlib import Path

        from build_knowledge import mark_papers_indexed
        from utils.dedup_manager import DedupManager

        with tempfile.TemporaryDirectory() as tmp:
            paper = Path(tmp) / "demo.pdf"
            paper.write_bytes(b"pdf")
            record_path = Path(tmp) / "records.json"

            mark_papers_indexed([str(paper)], str(record_path))

            records = DedupManager(str(record_path)).get_all_records()
            self.assertEqual(len(records), 1)
            self.assertEqual(next(iter(records.values()))["filename"], "demo.pdf")

    def test_prepare_chunks_for_chroma_serializes_list_metadata(self):
        from build_knowledge import prepare_chunks_for_chroma

        chunks = [
            Document(
                page_content="vision summary",
                metadata={
                    "block_type": "vision_summary",
                    "vision_trigger_reason": ["forced_page", "figure_dense_page"],
                    "quality_flags": ["vision_generated"],
                },
            )
        ]

        prepared = prepare_chunks_for_chroma(chunks)

        self.assertEqual(prepared[0].metadata["vision_trigger_reason"], '["forced_page", "figure_dense_page"]')
        self.assertEqual(prepared[0].metadata["quality_flags"], '["vision_generated"]')
        self.assertEqual(chunks[0].metadata["vision_trigger_reason"], ["forced_page", "figure_dense_page"])


if __name__ == "__main__":
    unittest.main()
