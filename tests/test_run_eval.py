import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from eval.run_eval import evaluate_results_file


class RunEvalTest(unittest.TestCase):
    def test_evaluate_results_file_writes_summary_report(self):
        rows = [
            {
                "id": "q001",
                "task_type": "definition",
                "difficulty": "easy",
                "gold_sources": [{"file": "paper.pdf", "page": 1}],
                "retrieved_sources": [
                    {"file": "noise.pdf", "page": 0},
                    {"file": "paper.pdf", "page": 1},
                ],
                "elapsed_sec": 2.5,
                "error": None,
                "skipped": False,
            },
            {
                "id": "q002",
                "task_type": "compare",
                "difficulty": "hard",
                "gold_sources": [
                    {"file": "a.pdf", "page": 1},
                    {"file": "b.pdf", "page": 2},
                ],
                "retrieved_sources": [{"file": "a.pdf", "page": 1}],
                "elapsed_sec": 4.0,
                "error": None,
                "skipped": False,
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "baseline.jsonl"
            output_dir = tmp_path / "reports"
            with input_path.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            report = evaluate_results_file(input_path, label="demo", output_dir=output_dir, k=2)

            self.assertEqual(report["label"], "demo")
            self.assertEqual(report["sample_count"], 2)
            self.assertEqual(report["averages"]["recall_at_2"], 0.75)
            self.assertEqual(report["averages"]["mrr"], 0.75)
            self.assertEqual(report["source_hit_counts"], {"full": 1, "partial": 1, "missing": 0})
            self.assertEqual(report["by_task_type"]["definition"]["sample_count"], 1)
            self.assertEqual(report["by_task_type"]["compare"]["avg_recall_at_2"], 0.5)
            self.assertEqual([item["id"] for item in report["low_recall_samples"]], ["q002"])
            self.assertTrue((output_dir / "report_demo.json").exists())

    def test_run_eval_script_can_be_executed_directly(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "baseline.jsonl"
            output_dir = tmp_path / "reports"
            input_path.write_text(
                json.dumps(
                    {
                        "id": "q001",
                        "task_type": "definition",
                        "difficulty": "easy",
                        "gold_sources": [{"file": "paper.pdf", "page": 1}],
                        "retrieved_sources": [{"file": "paper.pdf", "page": 1}],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "eval/run_eval.py",
                    "--input",
                    str(input_path),
                    "--label",
                    "direct",
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((output_dir / "report_direct.json").exists())


if __name__ == "__main__":
    unittest.main()
