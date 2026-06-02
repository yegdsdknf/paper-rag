import unittest

from eval.run_eval import evaluate_rows


class AgenticEvalMetricsTest(unittest.TestCase):
    def test_evaluate_rows_summarizes_agent_trace_metrics(self):
        rows = [
            {
                "id": "a1",
                "question": "GPT-3 使用 Transformer 结构的证据在哪一页？",
                "gold_sources": [{"file": "gpt3.pdf", "page": 7}],
                "gold_evidence": ["same model and architecture as GPT-2"],
                "predicted_answer": "证据在 gpt3.pdf 第 7 页。",
                "retrieved_sources": [{"file": "gpt3.pdf", "page": 7}],
                "elapsed_sec": 2.0,
                "agent_trace": {
                    "enabled": True,
                    "plan": [{"id": "g1"}, {"id": "g2"}],
                    "verification": [
                        {"goal_id": "g1", "status": "supported"},
                        {"goal_id": "g2", "status": "partial"},
                    ],
                    "repair_rounds": 1,
                    "repair_success": True,
                    "agent_elapsed_sec": 0.8,
                },
            },
            {
                "id": "a2",
                "question": "BERT 的全称是什么？",
                "gold_sources": [{"file": "bert.pdf", "page": 0}],
                "gold_evidence": ["Bidirectional Encoder Representations"],
                "predicted_answer": "BERT 的全称是 Bidirectional Encoder Representations from Transformers。",
                "retrieved_sources": [{"file": "bert.pdf", "page": 0}],
                "elapsed_sec": 1.0,
            },
        ]

        report = evaluate_rows(rows, label="agentic_demo", k=5)

        self.assertEqual(report["agent"]["enabled_count"], 1)
        self.assertEqual(report["agent"]["avg_evidence_goal_count"], 2.0)
        self.assertEqual(report["agent"]["goal_support_rate"], 0.5)
        self.assertEqual(report["agent"]["repair_trigger_rate"], 1.0)
        self.assertEqual(report["agent"]["repair_success_rate"], 1.0)
        self.assertEqual(report["agent"]["avg_agent_elapsed_sec"], 0.8)
