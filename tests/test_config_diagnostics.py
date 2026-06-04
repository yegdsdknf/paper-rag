import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class DiagnosticsReportTest(unittest.TestCase):
    def test_report_summary_exit_code_and_json_shape(self):
        from paper_rag.config.diagnostics import DiagnosticCheck, build_report, report_to_dict

        report = build_report(
            [
                DiagnosticCheck("config.load", "配置文件", "OK", "配置已加载", elapsed_sec=0.01),
                DiagnosticCheck(
                    "ollama.service",
                    "Ollama 服务",
                    "ERROR",
                    "无法连接 Ollama",
                    suggestion="运行 ollama serve",
                    elapsed_sec=0.02,
                ),
                DiagnosticCheck(
                    "index.manifest",
                    "索引 Manifest",
                    "WARN",
                    "未找到 index_manifest.json",
                    suggestion="重新运行 python main.py build",
                    elapsed_sec=0.0,
                ),
            ],
            settings_snapshot={"llm_model": "qwen2.5:3b"},
        )

        payload = report_to_dict(report)

        self.assertEqual(report.status, "ERROR")
        self.assertEqual(report.exit_code, 1)
        self.assertEqual(payload["summary"], {"OK": 1, "WARN": 1, "ERROR": 1})
        self.assertEqual(payload["settings"], {"llm_model": "qwen2.5:3b"})
        self.assertEqual(payload["checks"][1]["id"], "ollama.service")
        self.assertIn("elapsed_sec", payload["checks"][1])

    def test_render_text_puts_summary_and_blocking_issues_before_checks(self):
        from paper_rag.config.diagnostics import DiagnosticCheck, build_report, render_text

        report = build_report(
            [
                DiagnosticCheck("config.load", "配置文件", "OK", "配置已加载", elapsed_sec=0.01),
                DiagnosticCheck(
                    "ollama.service",
                    "Ollama 服务",
                    "ERROR",
                    "无法连接 Ollama",
                    suggestion="运行 ollama serve",
                    elapsed_sec=0.02,
                ),
            ],
            settings_snapshot={"persist_directory": "./chroma_db", "llm_model": "qwen2.5:3b"},
        )

        text = render_text(report)

        self.assertLess(text.index("Blocking issues"), text.index("Checks"))
        self.assertIn("Paper-RAG Doctor: ERROR", text)
        self.assertIn("运行 ollama serve", text)
        self.assertIn("persist_directory: ./chroma_db", text)


class DiagnosticsRunnerTest(unittest.TestCase):
    def test_config_load_failure_stops_after_single_error(self):
        from paper_rag.config import ConfigError
        from paper_rag.config.diagnostics import run_diagnostics

        report = run_diagnostics(config_loader=lambda: (_ for _ in ()).throw(ConfigError("缺少必要配置项")))

        self.assertEqual(report.status, "ERROR")
        self.assertEqual([check.id for check in report.checks], ["config.load"])
        self.assertIn("config.yaml", report.checks[0].suggestion)

    def test_missing_persist_directory_marks_chroma_as_skipped_warn(self):
        from paper_rag.config.diagnostics import run_diagnostics

        report = run_diagnostics(
            config_loader=lambda: {
                "persist_directory": "./missing_db",
                "collection_name": "langchain",
                "embedding_model": "local-model",
                "llm_model": "qwen2.5:3b",
                "temperature": 0.1,
                "k": 6,
                "chunk_size": 500,
                "chunk_overlap": 100,
                "separators": ["\n\n", "\n"],
            },
            path_exists=lambda path: False if Path(path).as_posix().endswith("missing_db") else True,
            chroma_count_loader=lambda _settings: 3,
            ollama_tags_loader=lambda _settings: ["qwen2.5:3b"],
            embedding_checker=lambda _settings: None,
            reranker_locator=lambda _settings: True,
            papers_pdf_counter=lambda _root: 1,
        )

        by_id = {check.id: check for check in report.checks}
        self.assertEqual(by_id["path.persist_directory"].status, "ERROR")
        self.assertEqual(by_id["chroma.collection"].status, "WARN")
        self.assertIn("跳过", by_id["chroma.collection"].message)

    def test_successful_lightweight_diagnostics_returns_fixed_checks(self):
        from paper_rag.config.diagnostics import run_diagnostics

        with TemporaryDirectory() as tmp:
            report = run_diagnostics(
                config_loader=lambda: {
                    "persist_directory": tmp,
                    "collection_name": "langchain",
                    "embedding_model": "local-model",
                    "llm_model": "qwen2.5:3b",
                    "temperature": 0.1,
                    "k": 6,
                    "chunk_size": 500,
                    "chunk_overlap": 100,
                    "separators": ["\n\n", "\n"],
                    "enable_rerank": False,
                },
                path_exists=lambda path: Path(path).exists(),
                manifest_loader=lambda _settings: {"chunk_schema_version": "v1"},
                chroma_count_loader=lambda _settings: 3,
                ollama_tags_loader=lambda _settings: ["qwen2.5:3b"],
                embedding_checker=lambda _settings: None,
                reranker_locator=lambda _settings: True,
                papers_pdf_counter=lambda _root: 2,
            )

        self.assertEqual(report.status, "OK")
        self.assertEqual(
            [check.id for check in report.checks],
            [
                "config.load",
                "path.persist_directory",
                "chroma.collection",
                "index.manifest",
                "ollama.service",
                "ollama.model",
                "embedding.model",
                "reranker.model",
                "path.papers",
            ],
        )
        self.assertEqual(report.exit_code, 0)

    def test_slow_diagnostics_adds_elapsed_warning(self):
        from paper_rag.config.diagnostics import run_diagnostics

        with TemporaryDirectory() as tmp:
            report = run_diagnostics(
                config_loader=lambda: {
                    "persist_directory": tmp,
                    "collection_name": "langchain",
                    "embedding_model": "local-model",
                    "llm_model": "qwen2.5:3b",
                    "temperature": 0.1,
                    "k": 6,
                    "chunk_size": 500,
                    "chunk_overlap": 100,
                    "separators": ["\n\n", "\n"],
                    "enable_rerank": False,
                },
                path_exists=lambda path: Path(path).exists(),
                manifest_loader=lambda _settings: {"chunk_schema_version": "v1"},
                chroma_count_loader=lambda _settings: 3,
                ollama_tags_loader=lambda _settings: ["qwen2.5:3b"],
                embedding_checker=lambda _settings: None,
                reranker_locator=lambda _settings: True,
                papers_pdf_counter=lambda _root: 2,
                slow_threshold_sec=0.0,
            )

        self.assertEqual(report.status, "WARN")
        self.assertIn("diagnostics.elapsed", [check.id for check in report.checks])


class DiagnosticsCliTest(unittest.TestCase):
    def test_render_json_is_parseable_without_extra_text(self):
        from paper_rag.config.diagnostics import DiagnosticCheck, build_report, render_json

        report = build_report(
            [DiagnosticCheck("config.load", "配置文件", "OK", "配置已加载")],
            settings_snapshot={},
        )

        payload = json.loads(render_json(report))
        self.assertEqual(payload["status"], "OK")
        self.assertEqual(payload["checks"][0]["id"], "config.load")


if __name__ == "__main__":
    unittest.main()
