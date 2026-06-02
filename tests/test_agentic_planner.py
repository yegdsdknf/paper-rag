import unittest

from paper_rag.agentic.planner import plan_evidence_goals


class _Message:
    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    def __init__(self, content: str):
        self.content = content
        self.prompts: list[str] = []

    def invoke(self, prompt: str):
        self.prompts.append(prompt)
        return _Message(self.content)


class BrokenLLM:
    def invoke(self, prompt: str):
        raise RuntimeError("llm unavailable")


class AgenticPlannerTest(unittest.TestCase):
    def test_uses_valid_llm_goals_and_limits_to_four(self):
        llm = FakeLLM(
            """
            {
              "goals": [
                {"goal_type": "page_evidence", "claim": "c1", "query": "q1", "source_hint": "paper-a.pdf", "page_hint": 3},
                {"goal_type": "figure_evidence", "claim": "c2", "query": "q2", "source_hint": "paper-b.pdf", "page_hint": "7"},
                {"goal_type": "method_overview", "claim": "c3", "query": "q3", "source_hint": "paper-a.pdf"},
                {"goal_type": "compare_dimension", "claim": "c4", "query": "q4", "source_hint": "paper-b.pdf"},
                {"goal_type": "page_evidence", "claim": "c5", "query": "q5", "source_hint": "paper-a.pdf"}
              ]
            }
            """
        )

        goals = plan_evidence_goals(
            question="原始问题",
            standalone_question="比较两篇论文的方法",
            source_hints=["paper-a.pdf", "paper-b.pdf"],
            task_type="compare",
            llm=llm,
        )

        self.assertEqual(4, len(goals))
        self.assertEqual("page_evidence", goals[0]["goal_type"])
        self.assertEqual("paper-a.pdf", goals[0]["source_hint"])
        self.assertEqual(3, goals[0]["page_hint"])
        self.assertEqual("figure_evidence", goals[1]["goal_type"])
        self.assertEqual(7, goals[1]["page_hint"])
        self.assertIn("原始问题", llm.prompts[0])
        self.assertIn("paper-a.pdf", llm.prompts[0])

    def test_broken_llm_compare_fallback_creates_one_goal_per_source_up_to_four(self):
        goals = plan_evidence_goals(
            question="q",
            standalone_question="比较这些论文",
            source_hints=["a.pdf", "b.pdf", "c.pdf", "d.pdf", "e.pdf"],
            task_type="compare",
            llm=BrokenLLM(),
        )

        self.assertEqual(4, len(goals))
        self.assertEqual(["a.pdf", "b.pdf", "c.pdf", "d.pdf"], [goal["source_hint"] for goal in goals])
        self.assertTrue(all(goal["goal_type"] == "compare_dimension" for goal in goals))
        self.assertTrue(all("比较这些论文" in goal["query"] for goal in goals))

    def test_rule_fallback_types_for_single_goal_tasks(self):
        cases = [
            ("figure", "figure_evidence"),
            ("evidence", "page_evidence"),
            ("method", "method_overview"),
        ]

        for task_type, expected_goal_type in cases:
            with self.subTest(task_type=task_type):
                goals = plan_evidence_goals(
                    question="q",
                    standalone_question="独立问题",
                    source_hints=["main.pdf"],
                    task_type=task_type,
                )

                self.assertEqual(1, len(goals))
                self.assertEqual(expected_goal_type, goals[0]["goal_type"])
                self.assertEqual("main.pdf", goals[0]["source_hint"])
                self.assertIsNone(goals[0]["page_hint"])

    def test_normalizes_unknown_llm_goal_type_and_source_hint(self):
        llm = FakeLLM(
            """
            {
              "goals": [
                {"goal_type": "invented", "claim": "x", "query": "x query", "source_hint": "unknown.pdf", "page_hint": "bad"}
              ]
            }
            """
        )

        goals = plan_evidence_goals(
            question="q",
            standalone_question="独立问题",
            source_hints=["known.pdf"],
            task_type="evidence",
            llm=llm,
        )

        self.assertEqual("method_overview", goals[0]["goal_type"])
        self.assertEqual("", goals[0]["source_hint"])
        self.assertIsNone(goals[0]["page_hint"])

    def test_parses_noisy_fenced_json_from_llm(self):
        llm = FakeLLM(
            """
            我先给出规划：
            ```json
            {
              "goals": [
                {"goal_type": "figure_evidence", "claim": "find figure", "query": "figure query", "source_hint": "paper.pdf"}
              ]
            }
            ```
            """
        )

        goals = plan_evidence_goals(
            question="q",
            standalone_question="找图",
            source_hints=["paper.pdf"],
            task_type="figure",
            llm=llm,
        )

        self.assertEqual(1, len(goals))
        self.assertEqual("figure_evidence", goals[0]["goal_type"])
        self.assertEqual("paper.pdf", goals[0]["source_hint"])


if __name__ == "__main__":
    unittest.main()
