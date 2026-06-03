import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document

import rag_pipeline
from query_logger import build_query_log_record, save_query_log_record


class FakeConversation:
    history = []

    def reformulate(self, question):
        return question

    def format_history(self):
        return ""


class FakeStreamingLLM:
    def stream(self, prompt):
        yield type("Chunk", (), {"content": "ans"})()
        yield type("Chunk", (), {"content": "wer"})()


class QueryLoggerTest(unittest.TestCase):
    def test_build_query_log_record_normalizes_sources_and_timings(self):
        docs = [
            Document(
                page_content="evidence text",
                metadata={"source": "./papers/bert.pdf", "page": 2, "rerank_score": 0.9},
            )
        ]

        record = build_query_log_record(
            question="BERT 是什么？",
            standalone_question="BERT 是什么？",
            route="mixed",
            llm_model="qwen2.5:3b",
            embedding_device="cuda",
            index_version="idx_config123",
            feature_flags={
                "rerank": True,
                "query_expansion": False,
                "context_compression": True,
                "parent_retrieval": False,
                "agentic_query": True,
            },
            docs=docs,
            elapsed={"rewrite": 0.01, "retrieve": 0.2, "generate": 1.0, "total": 1.21},
            query_variants=["bert definition"],
            context_stats={"input_chars": 100, "output_chars": 60},
            agent_trace={"enabled": True, "repair_rounds": 1},
        )

        self.assertEqual(record["question"], "BERT 是什么？")
        self.assertEqual(record["route"], "mixed")
        self.assertEqual(record["query_variants"], ["bert definition"])
        self.assertEqual(record["index_version"], "idx_config123")
        self.assertEqual(
            record["feature_flags"],
            {
                "rerank": True,
                "query_expansion": False,
                "context_compression": True,
                "parent_retrieval": False,
                "agentic_query": True,
            },
        )
        self.assertEqual(record["retrieved_sources"][0]["file"], "bert.pdf")
        self.assertEqual(record["retrieved_sources"][0]["page"], 2)
        self.assertEqual(record["retrieved_sources"][0]["rerank_score"], 0.9)
        self.assertEqual(record["elapsed"]["total"], 1.21)
        self.assertEqual(record["context"]["output_chars"], 60)
        self.assertEqual(record["agent_trace"]["repair_rounds"], 1)
        self.assertIn("timestamp", record)

    def test_save_query_log_record_appends_jsonl(self):
        record = build_query_log_record(
            question="q",
            standalone_question="q",
            route="hyde",
            llm_model="qwen2.5:3b",
            embedding_device="cpu",
            docs=[],
            elapsed={"total": 0.1},
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "logs" / "query_runs.jsonl"
            saved = save_query_log_record(record, path)

            self.assertEqual(saved, path)
            rows = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 1)
            self.assertEqual(json.loads(rows[0])["route"], "hyde")
            self.assertEqual(json.loads(rows[0])["agent_trace"], {})

    def test_ask_with_context_writes_query_log_when_enabled(self):
        docs = [Document(page_content="answer evidence", metadata={"source": "paper.pdf", "page": 1})]

        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "query_runs.jsonl"
            with (
                patch.dict(
                    rag_pipeline.config,
                    {
                        "enable_query_logging": True,
                        "query_log_path": str(log_path),
                    },
                ),
                patch.object(rag_pipeline, "_route_retrieve", return_value=(docs, "mixed")),
                patch.object(rag_pipeline, "_generate_answer", return_value="answer"),
                patch.object(rag_pipeline, "_get_embedding_device", return_value="cuda"),
            ):
                answer, returned_docs = rag_pipeline.ask_with_context(
                    hybrid=object(),
                    conversation=FakeConversation(),
                    question="问题",
                    llm_model="qwen2.5:3b",
                )

            self.assertEqual(answer, "answer")
            self.assertEqual(returned_docs, docs)
            record = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(record["question"], "问题")
            self.assertEqual(record["route"], "mixed")
            self.assertEqual(record["llm_model"], "qwen2.5:3b")
            self.assertEqual(record["embedding_device"], "cuda")
            self.assertEqual(record["retrieved_sources"][0]["file"], "paper.pdf")
            self.assertGreaterEqual(record["elapsed"]["total"], 0)

    def test_ask_stream_writes_query_log_after_generation(self):
        docs = [Document(page_content="stream evidence", metadata={"source": "paper.pdf", "page": 3})]

        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "stream_runs.jsonl"
            with (
                patch.dict(
                    rag_pipeline.config,
                    {
                        "enable_query_logging": True,
                        "query_log_path": str(log_path),
                        "enable_context_compression": False,
                        "enable_parent_retrieval": False,
                    },
                ),
                patch.object(rag_pipeline, "_route_retrieve", return_value=(docs, "hyde")),
                patch.object(rag_pipeline, "_get_llm", return_value=FakeStreamingLLM()),
                patch.object(rag_pipeline, "_get_embedding_device", return_value="cpu"),
            ):
                events = list(
                    rag_pipeline.ask_stream(
                        hybrid=object(),
                        conversation=FakeConversation(),
                        question="流式问题",
                        llm_model="qwen2.5:3b",
                    )
                )

            self.assertEqual("".join(event["data"] for event in events if event["type"] == "token"), "answer")
            record = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(record["question"], "流式问题")
            self.assertEqual(record["route"], "hyde")
            self.assertEqual(record["retrieved_sources"][0]["page"], 3)
            self.assertIn("generate", record["elapsed"])

    def test_ask_stream_writes_error_log_when_llm_unavailable(self):
        docs = [Document(page_content="stream evidence", metadata={"source": "paper.pdf", "page": 3})]

        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "stream_errors.jsonl"
            with (
                patch.dict(
                    rag_pipeline.config,
                    {
                        "enable_query_logging": True,
                        "query_log_path": str(log_path),
                        "enable_context_compression": False,
                        "enable_parent_retrieval": False,
                    },
                ),
                patch.object(rag_pipeline, "_route_retrieve", return_value=(docs, "hyde")),
                patch.object(rag_pipeline, "_get_llm", return_value=None),
                patch.object(rag_pipeline, "_get_embedding_device", return_value="cpu"),
            ):
                events = list(
                    rag_pipeline.ask_stream(
                        hybrid=object(),
                        conversation=FakeConversation(),
                        question="流式问题",
                        llm_model="qwen2.5:3b",
                    )
                )

            self.assertEqual(events[-1]["data"], "❌ LLM 模型未连接")
            record = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(record["error"], "LLM 模型未连接")
            self.assertEqual(record["retrieved_sources"][0]["file"], "paper.pdf")


if __name__ == "__main__":
    unittest.main()
