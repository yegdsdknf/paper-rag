import tempfile
import unittest
from pathlib import Path

from langchain_core.documents import Document

from app_services import (
    build_feedback_payload,
    collect_stream_answer,
    save_feedback_from_payload,
    save_uploaded_pdfs,
)


class FakeUpload:
    def __init__(self, name, data):
        self.name = name
        self._data = data

    def getbuffer(self):
        return self._data


class AppServicesTest(unittest.TestCase):
    def test_save_uploaded_pdfs_writes_files_and_returns_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved = save_uploaded_pdfs(
                [FakeUpload("a.pdf", b"pdf-a"), FakeUpload("b.pdf", b"pdf-b")],
                Path(tmp),
            )

            self.assertEqual([path.name for path in saved], ["a.pdf", "b.pdf"])
            self.assertEqual((Path(tmp) / "a.pdf").read_bytes(), b"pdf-a")
            self.assertEqual((Path(tmp) / "b.pdf").read_bytes(), b"pdf-b")

    def test_collect_stream_answer_returns_answer_docs_route_and_rewrite(self):
        docs = [Document(page_content="evidence", metadata={"source": "paper.pdf", "page": 1})]

        def fake_ask_stream(hybrid, conversation, question, llm_model, temperature):
            yield {"type": "rewrite", "data": "standalone question"}
            yield {"type": "route", "data": "mixed"}
            yield {"type": "docs", "data": docs}
            yield {"type": "token", "data": "ans"}
            yield {"type": "token", "data": "wer"}

        result = collect_stream_answer(
            ask_stream_fn=fake_ask_stream,
            hybrid=object(),
            conversation=object(),
            question="question",
            llm_model="qwen2.5:3b",
            temperature=0.1,
        )

        self.assertEqual(result.answer, "answer")
        self.assertEqual(result.docs, docs)
        self.assertEqual(result.route, "mixed")
        self.assertEqual(result.rewrite, "standalone question")

    def test_build_feedback_payload_preserves_fields(self):
        docs = [Document(page_content="evidence", metadata={"source": "paper.pdf", "page": 1})]

        payload = build_feedback_payload(
            question="q",
            answer="a",
            docs=docs,
            route="hyde",
            llm_model="qwen2.5:3b",
        )

        self.assertEqual(payload["question"], "q")
        self.assertEqual(payload["answer"], "a")
        self.assertEqual(payload["sources"], docs)
        self.assertEqual(payload["route"], "hyde")
        self.assertEqual(payload["llm_model"], "qwen2.5:3b")

    def test_save_feedback_from_payload_rejects_blank_note(self):
        with self.assertRaises(ValueError):
            save_feedback_from_payload({"question": "q", "answer": "a", "sources": []}, " ")


if __name__ == "__main__":
    unittest.main()
