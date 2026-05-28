import unittest

from eval.metrics import (
    mrr,
    recall_at_k,
    source_hit_status,
    source_matches,
)


class EvalMetricsTest(unittest.TestCase):
    def test_source_matches_normalizes_file_paths_and_page_numbers(self):
        retrieved = {"file": r"papers\attention is all you need.pdf", "page": "5"}
        gold = {"file": "attention is all you need.pdf", "page": 5}

        self.assertTrue(source_matches(retrieved, gold))

    def test_recall_at_k_counts_unique_gold_sources_found_in_top_k(self):
        retrieved = [
            {"file": "wrong.pdf", "page": 0},
            {"file": "bert.pdf", "page": 1},
            {"file": "bert.pdf", "page": 3},
        ]
        gold = [
            {"file": "bert.pdf", "page": 1},
            {"file": "bert.pdf", "page": 3},
        ]

        self.assertEqual(recall_at_k(retrieved, gold, k=2), 0.5)
        self.assertEqual(recall_at_k(retrieved, gold, k=3), 1.0)

    def test_mrr_returns_first_relevant_rank_reciprocal(self):
        retrieved = [
            {"file": "wrong.pdf", "page": 0},
            {"file": "attention is all you need.pdf", "page": 7},
        ]
        gold = [{"file": "attention is all you need.pdf", "page": 7}]

        self.assertEqual(mrr(retrieved, gold), 0.5)

    def test_source_hit_status_distinguishes_full_partial_and_missing(self):
        gold = [
            {"file": "bert.pdf", "page": 1},
            {"file": "bert.pdf", "page": 3},
        ]

        self.assertEqual(source_hit_status([], gold), "missing")
        self.assertEqual(source_hit_status([{"file": "bert.pdf", "page": 1}], gold), "partial")
        self.assertEqual(source_hit_status(gold, gold), "full")


if __name__ == "__main__":
    unittest.main()
