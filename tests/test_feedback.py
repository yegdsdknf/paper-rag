import json
import tempfile
import unittest
from pathlib import Path

from langchain_core.documents import Document

from feedback import build_feedback_record, save_feedback_record


class FeedbackTest(unittest.TestCase):
    def test_build_feedback_record_preserves_question_answer_sources_and_note(self):
        docs = [
            Document(
                page_content="The Transformer is based solely on attention mechanisms.",
                metadata={"source": "./papers/attention is all you need.pdf", "page": 0},
            )
        ]

        record = build_feedback_record(
            question="Transformer 是基于什么机制构建的？",
            answer="基于注意力机制。",
            sources=docs,
            note="回答正确，但证据页码需要确认。",
            route="mixed_multi_query",
            llm_model="qwen2.5:3b",
        )

        self.assertEqual(record["question"], "Transformer 是基于什么机制构建的？")
        self.assertEqual(record["answer"], "基于注意力机制。")
        self.assertEqual(record["note"], "回答正确，但证据页码需要确认。")
        self.assertEqual(record["route"], "mixed_multi_query")
        self.assertEqual(record["llm_model"], "qwen2.5:3b")
        self.assertEqual(record["sources"][0]["file"], "attention is all you need.pdf")
        self.assertEqual(record["sources"][0]["page"], 0)
        self.assertIn("attention mechanisms", record["sources"][0]["content_preview"])
        self.assertIn("timestamp", record)

    def test_save_feedback_record_appends_jsonl_and_creates_directory(self):
        record = build_feedback_record(
            question="q",
            answer="a",
            sources=[],
            note="needs benchmark",
            route="hyde",
            llm_model="qwen2.5:3b",
        )

        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "feedback" / "feedback.jsonl"

            saved_path = save_feedback_record(record, output_path)

            self.assertEqual(saved_path, output_path)
            rows = output_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 1)
            self.assertEqual(json.loads(rows[0])["note"], "needs benchmark")


if __name__ == "__main__":
    unittest.main()
