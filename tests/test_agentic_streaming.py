import unittest
from unittest.mock import patch

from langchain_core.documents import Document

import rag_pipeline
from paper_rag.ui.services import collect_stream_answer


class FakeConversation:
    history = []

    def reformulate(self, question):
        return question

    def format_history(self):
        return ""


class AgenticStreamingTest(unittest.TestCase):
    def test_ask_stream_agentic_yields_status_route_docs_token_and_trace(self):
        docs = [Document(page_content="agent evidence", metadata={"source": "paper.pdf", "page": 4})]
        trace = {"enabled": True, "repair_rounds": 1}

        with (
            patch.dict(
                rag_pipeline.config,
                {
                    "enable_agentic_query": False,
                    "agent_debug_trace": True,
                    "enable_query_logging": False,
                },
            ),
            patch.object(
                rag_pipeline,
                "run_agentic_rag",
                return_value={
                    "final_docs": docs,
                    "route": "agentic_mixed",
                    "agent_trace": trace,
                    "verified_summary": "verified",
                },
            ),
            patch.object(rag_pipeline, "mentioned_source_files", return_value=["paper.pdf"]),
            patch.object(rag_pipeline, "_get_llm", return_value=object()),
            patch.object(rag_pipeline, "stream_answer_tokens", return_value=iter(["ans", "wer"])),
        ):
            events = list(
                rag_pipeline.ask_stream(
                    hybrid=object(),
                    conversation=FakeConversation(),
                    question="问题",
                    force_agent=True,
                )
            )

        self.assertEqual(
            [event["data"] for event in events if event["type"] == "agent_status"],
            ["正在拆分证据目标...", "正在校验证据并组装上下文..."],
        )
        self.assertEqual([event["data"] for event in events if event["type"] == "agent_trace"], [trace])
        self.assertEqual([event["data"] for event in events if event["type"] == "route"], ["agentic_mixed"])
        self.assertEqual([event["data"] for event in events if event["type"] == "docs"], [docs])
        self.assertEqual("".join(event["data"] for event in events if event["type"] == "token"), "answer")

    def test_ask_stream_agentic_omits_trace_when_debug_disabled(self):
        docs = [Document(page_content="agent evidence", metadata={"source": "paper.pdf", "page": 4})]

        with (
            patch.dict(
                rag_pipeline.config,
                {
                    "enable_agentic_query": False,
                    "agent_debug_trace": False,
                    "enable_query_logging": False,
                },
            ),
            patch.object(
                rag_pipeline,
                "run_agentic_rag",
                return_value={
                    "final_docs": docs,
                    "route": "agentic_mixed",
                    "agent_trace": {"enabled": True},
                    "verified_summary": "",
                },
            ),
            patch.object(rag_pipeline, "_get_llm", return_value=object()),
            patch.object(rag_pipeline, "stream_answer_tokens", return_value=iter(["ok"])),
        ):
            events = list(
                rag_pipeline.ask_stream(
                    hybrid=object(),
                    conversation=FakeConversation(),
                    question="问题",
                    force_agent=True,
                )
            )

        self.assertEqual([event for event in events if event["type"] == "agent_trace"], [])

    def test_collect_stream_answer_forwards_force_agent_and_ignores_agent_events(self):
        docs = [Document(page_content="evidence", metadata={"source": "paper.pdf", "page": 1})]
        calls = []

        def fake_ask_stream(hybrid, conversation, question, llm_model, temperature, force_agent=None):
            calls.append(force_agent)
            yield {"type": "agent_status", "data": "正在拆分证据目标..."}
            yield {"type": "agent_trace", "data": {"enabled": True}}
            yield {"type": "rewrite", "data": "standalone"}
            yield {"type": "route", "data": "agentic_mixed"}
            yield {"type": "docs", "data": docs}
            yield {"type": "token", "data": "ans"}
            yield {"type": "token", "data": "wer"}

        result = collect_stream_answer(
            ask_stream_fn=fake_ask_stream,
            hybrid=object(),
            conversation=object(),
            question="question",
            llm_model="qwen2.5:3b",
            temperature=0.1,
            force_agent=True,
        )

        self.assertEqual(calls, [True])
        self.assertEqual(result.answer, "answer")
        self.assertEqual(result.docs, docs)
        self.assertEqual(result.route, "agentic_mixed")
        self.assertEqual(result.rewrite, "standalone")


if __name__ == "__main__":
    unittest.main()
