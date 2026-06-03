import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks import run_baseline


class AgenticCliBenchmarkTest(unittest.TestCase):
    def test_parse_args_accepts_agent_with_explicit_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "agent.jsonl"

            args = run_baseline.parse_args(["--agent", "--output", str(output)])

        self.assertTrue(args.agent)
        self.assertEqual(args.output, output)

    def test_parse_args_accepts_no_agent(self):
        args = run_baseline.parse_args(["--no-agent"])

        self.assertFalse(args.agent)
        self.assertEqual(args.output, run_baseline.DEFAULT_OUTPUT_PATH)

    def test_parse_args_agent_uses_agentic_default_output(self):
        args = run_baseline.parse_args(["--agent"])

        self.assertTrue(args.agent)
        self.assertEqual(args.output, run_baseline.DEFAULT_AGENT_OUTPUT_PATH)

    def test_run_sample_forwards_force_agent_to_rag_pipeline(self):
        sample = {
            "id": "s1",
            "question": "q",
            "task_type": "evidence",
            "difficulty": "easy",
            "gold_answer": "gold",
            "gold_sources": [],
        }
        calls = []

        class FakeConversation:
            def __init__(self, **kwargs):
                self.history = []

        class FakeRagPipeline:
            @staticmethod
            def ask_with_context(*args, **kwargs):
                calls.append(kwargs)
                return "answer", []

        result = run_baseline._run_sample(
            sample,
            hybrid=object(),
            conversation_cls=FakeConversation,
            rag_pipeline=FakeRagPipeline,
            config={"temperature": 0.1},
            llm_model="qwen2.5:3b",
            source_serializer=lambda docs, preview_chars: [],
            force_agent=True,
        )

        self.assertIsNone(result["error"])
        self.assertTrue(calls[0]["force_agent"])

    def test_run_sample_omits_force_agent_for_legacy_pipeline_when_unset(self):
        sample = {
            "id": "s1",
            "question": "q",
            "task_type": "evidence",
            "difficulty": "easy",
            "gold_answer": "gold",
            "gold_sources": [],
        }

        class FakeConversation:
            def __init__(self, **kwargs):
                self.history = []

        class FakeRagPipeline:
            @staticmethod
            def ask_with_context(hybrid, conversation, question, llm_model, temperature):
                return "answer", []

        result = run_baseline._run_sample(
            sample,
            hybrid=object(),
            conversation_cls=FakeConversation,
            rag_pipeline=FakeRagPipeline,
            config={"temperature": 0.1},
            llm_model="qwen2.5:3b",
            source_serializer=lambda docs, preview_chars: [],
            force_agent=None,
        )

        self.assertIsNone(result["error"])
        self.assertEqual(result["predicted_answer"], "answer")

    def test_run_sample_uses_trace_helper_when_available(self):
        sample = {
            "id": "s1",
            "question": "q",
            "task_type": "evidence",
            "difficulty": "easy",
            "gold_answer": "gold",
            "gold_sources": [],
        }
        agent_trace = {"enabled": True, "repair_rounds": 1}
        calls = []

        class FakeConversation:
            def __init__(self, **kwargs):
                self.history = []

        class FakeRagPipeline:
            @staticmethod
            def ask_with_context_trace(*args, **kwargs):
                calls.append(kwargs)
                return "answer", [], agent_trace

        result = run_baseline._run_sample(
            sample,
            hybrid=object(),
            conversation_cls=FakeConversation,
            rag_pipeline=FakeRagPipeline,
            config={"temperature": 0.1},
            llm_model="qwen2.5:3b",
            source_serializer=lambda docs, preview_chars: [],
            force_agent=True,
        )

        self.assertIsNone(result["error"])
        self.assertTrue(calls[0]["force_agent"])
        self.assertEqual(result["agent_trace"], agent_trace)

    def test_query_main_forwards_agent_flag_to_ask_with_context(self):
        import query

        calls = []

        class FakeConversation:
            def __init__(self, *args, **kwargs):
                self.history = []

            def add_turn(self, question, answer):
                self.history.append((question, answer))

            def clear(self):
                self.history.clear()

        def fake_ask_with_context(*args, **kwargs):
            calls.append(kwargs)
            return "answer", []

        with (
            patch.object(query, "build_hybrid_retriever", return_value=object()),
            patch.object(query, "ConversationManager", FakeConversation),
            patch.object(query, "ask_with_context", fake_ask_with_context),
            patch("builtins.input", side_effect=["1", "问题", "q"]),
            patch.dict(query.config, {"llm_model": "qwen2.5:3b", "temperature": 0.1}),
        ):
            query.main(["--agent"])

        self.assertTrue(calls[0]["force_agent"])


if __name__ == "__main__":
    unittest.main()
