import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "benchmarks" / "benchmark_v1.jsonl"
HOLDOUT_PATH = ROOT / "benchmarks" / "holdout_v1.jsonl"


def _load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class BenchmarkHoldoutTest(unittest.TestCase):
    def test_holdout_is_disjoint_and_has_required_fields(self):
        train = _load_jsonl(TRAIN_PATH)
        holdout = _load_jsonl(HOLDOUT_PATH)
        train_ids = {row["id"] for row in train}
        train_questions = {row["question"] for row in train}

        self.assertGreaterEqual(len(holdout), 12)
        for row in holdout:
            self.assertNotIn(row["id"], train_ids)
            self.assertNotIn(row["question"], train_questions)
            for field in ["id", "question", "gold_answer", "gold_sources", "gold_evidence", "task_type", "difficulty"]:
                self.assertIn(field, row)
            self.assertTrue(row["gold_sources"])
            self.assertTrue(row["gold_evidence"])


if __name__ == "__main__":
    unittest.main()
