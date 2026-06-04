import json
import tempfile
import unittest
from pathlib import Path


def _source(file="paper.pdf", page=1):
    return {"file": file, "page": page}


def _row(sample_id, retrieved, elapsed=1.0, answer="contains evidence", gold_evidence=None):
    return {
        "id": sample_id,
        "question": f"q-{sample_id}",
        "gold_sources": [_source()],
        "retrieved_sources": retrieved,
        "gold_evidence": gold_evidence or ["evidence"],
        "predicted_answer": answer,
        "elapsed_sec": elapsed,
        "error": None,
        "skipped": False,
    }


def _write_jsonl(path: Path, rows: list[dict]):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class BenchmarkRegressionCheckTest(unittest.TestCase):
    def test_hard_retrieval_regression_returns_exit_code_1(self):
        from benchmarks.regression_check import compare_result_files

        with tempfile.TemporaryDirectory() as tmp:
            baseline = Path(tmp) / "baseline.jsonl"
            new = Path(tmp) / "new.jsonl"
            _write_jsonl(baseline, [_row("q1", [_source()], elapsed=1.0)])
            _write_jsonl(new, [_row("q1", [], elapsed=1.0)])

            report = compare_result_files(baseline, new)

        self.assertEqual(report.exit_code, 1)
        self.assertTrue(report.hard_failures)
        self.assertIn("missing_rate", [item.metric for item in report.hard_failures])

    def test_warn_only_regression_keeps_exit_code_0(self):
        from benchmarks.regression_check import compare_result_files

        with tempfile.TemporaryDirectory() as tmp:
            baseline = Path(tmp) / "baseline.jsonl"
            new = Path(tmp) / "new.jsonl"
            _write_jsonl(baseline, [_row("q1", [_source()], elapsed=1.0, answer="contains evidence")])
            _write_jsonl(new, [_row("q1", [_source()], elapsed=2.0, answer="missing")])

            report = compare_result_files(baseline, new)

        self.assertEqual(report.exit_code, 0)
        self.assertFalse(report.hard_failures)
        self.assertTrue(report.warnings)
        self.assertIn("avg_elapsed_sec", [item.metric for item in report.warnings])

    def test_mismatched_sample_ids_returns_input_error(self):
        from benchmarks.regression_check import compare_result_files

        with tempfile.TemporaryDirectory() as tmp:
            baseline = Path(tmp) / "baseline.jsonl"
            new = Path(tmp) / "new.jsonl"
            _write_jsonl(baseline, [_row("q1", [_source()])])
            _write_jsonl(new, [_row("q2", [_source()])])

            report = compare_result_files(baseline, new)

        self.assertEqual(report.exit_code, 2)
        self.assertEqual(report.status, "ERROR")
        self.assertIn("样本 id 不一致", report.message)

    def test_json_report_is_parseable(self):
        from benchmarks.regression_check import compare_result_files, render_json

        with tempfile.TemporaryDirectory() as tmp:
            baseline = Path(tmp) / "baseline.jsonl"
            new = Path(tmp) / "new.jsonl"
            _write_jsonl(baseline, [_row("q1", [_source()])])
            _write_jsonl(new, [_row("q1", [_source()])])

            report = compare_result_files(baseline, new)
            payload = json.loads(render_json(report))

        self.assertEqual(payload["status"], "OK")
        self.assertEqual(payload["exit_code"], 0)
        self.assertIn("metrics", payload)

    def test_cli_json_output_is_parseable(self):
        import io
        import sys
        from contextlib import redirect_stdout
        from unittest.mock import patch

        from benchmarks.regression_check import main

        with tempfile.TemporaryDirectory() as tmp:
            baseline = Path(tmp) / "baseline.jsonl"
            new = Path(tmp) / "new.jsonl"
            _write_jsonl(baseline, [_row("q1", [_source()])])
            _write_jsonl(new, [_row("q1", [_source()])])

            stdout = io.StringIO()
            with patch.object(
                sys,
                "argv",
                [
                    "regression_check.py",
                    "--baseline",
                    str(baseline),
                    "--current",
                    str(new),
                    "--json",
                ],
            ), redirect_stdout(stdout):
                exit_code = main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(stdout.getvalue())["status"], "OK")


if __name__ == "__main__":
    unittest.main()
