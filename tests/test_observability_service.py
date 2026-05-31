import json
import tempfile
import unittest
from pathlib import Path

from langchain_core.documents import Document

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


class QueryLogServiceTest(unittest.TestCase):
    def test_write_query_log_skips_when_disabled(self):
        from paper_rag.observability.service import write_query_log

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "query_runs.jsonl"
            result = write_query_log(
                settings=make_settings(enable_query_logging=False, query_log_path=str(path)),
                question="问题",
                standalone_question="问题",
                route="mixed",
                llm_model="qwen2.5:3b",
                docs=[],
                elapsed={"total": 0.1},
                embedding_device_fn=lambda: "cpu",
            )

            self.assertIsNone(result)
            self.assertFalse(path.exists())

    def test_write_query_log_builds_record_and_appends_jsonl(self):
        from paper_rag.observability.service import write_query_log

        docs = [Document(page_content="evidence", metadata={"source": "paper.pdf", "page": 2})]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "query_runs.jsonl"
            persist_dir = Path(tmp) / "chroma_db"
            persist_dir.mkdir()
            (persist_dir / "index_manifest.json").write_text(
                json.dumps({"index_version": "idx_manifest456"}, ensure_ascii=False),
                encoding="utf-8",
            )
            result = write_query_log(
                settings=make_settings(
                    persist_directory=str(persist_dir),
                    enable_query_logging=True,
                    query_log_path=str(path),
                    enable_rerank=True,
                    enable_query_expansion=False,
                    enable_context_compression=True,
                    enable_parent_retrieval=False,
                ),
                question="问题",
                standalone_question="独立问题",
                route="hyde",
                llm_model="qwen2.5:3b",
                docs=docs,
                elapsed={"retrieve": 0.2, "total": 0.3},
                embedding_device_fn=lambda: "cuda",
                context_stats={"input_chars": 8, "output_chars": 8},
                error="LLM 模型未连接",
            )

            self.assertEqual(result, path)
            record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(record["standalone_question"], "独立问题")
            self.assertEqual(record["route"], "hyde")
            self.assertEqual(record["embedding_device"], "cuda")
            self.assertEqual(record["index_version"], "idx_manifest456")
            self.assertEqual(
                record["feature_flags"],
                {
                    "rerank": True,
                    "query_expansion": False,
                    "context_compression": True,
                    "parent_retrieval": False,
                },
            )
            self.assertEqual(record["retrieved_sources"][0]["file"], "paper.pdf")
            self.assertEqual(record["context"]["input_chars"], 8)
            self.assertEqual(record["error"], "LLM 模型未连接")


if __name__ == "__main__":
    unittest.main()
