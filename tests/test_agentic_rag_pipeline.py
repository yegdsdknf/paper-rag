import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document

import rag_pipeline


class FakeConversation:
    history = []

    def reformulate(self, question):
        return question

    def format_history(self):
        return ""


class AgenticRagPipelineTest(unittest.TestCase):
    def test_ask_with_context_uses_agentic_docs_summary_and_trace_log(self):
        agent_docs = [
            Document(
                page_content="agent evidence",
                metadata={"source": "agent-paper.pdf", "page": 7},
            )
        ]
        agent_trace = {
            "enabled": True,
            "plan": [{"id": "g1", "goal_type": "page_evidence"}],
            "repair_rounds": 1,
        }

        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "query_runs.jsonl"
            with (
                patch.dict(
                    rag_pipeline.config,
                    {
                        "enable_agentic_query": True,
                        "agent_auto_for_complex": True,
                        "agent_max_repair_rounds": 2,
                        "agent_planner_model": "planner-model",
                        "agent_verifier_model": "verifier-model",
                        "agent_verifier_temperature": 0.0,
                        "enable_query_logging": True,
                        "query_log_path": str(log_path),
                    },
                ),
                patch.object(rag_pipeline, "mentioned_source_files", return_value=["agent-paper.pdf"]) as source_hints,
                patch.object(
                    rag_pipeline,
                    "run_agentic_rag",
                    return_value={
                        "final_docs": agent_docs,
                        "route": "agentic_mixed",
                        "agent_trace": agent_trace,
                        "verified_summary": "verified evidence summary",
                    },
                ) as run_agentic,
                patch.object(rag_pipeline, "_route_retrieve") as route_retrieve,
                patch.object(rag_pipeline, "_generate_answer", return_value="agent answer") as generate_answer,
                patch.object(rag_pipeline, "_get_llm", return_value=object()),
                patch.object(rag_pipeline, "_get_embedding_device", return_value="cpu"),
            ):
                answer, sources = rag_pipeline.ask_with_context(
                    hybrid=object(),
                    conversation=FakeConversation(),
                    question="请给出 agent-paper 这篇论文的证据在哪一页？",
                    llm_model="answer-model",
                    temperature=0.2,
                )

            self.assertEqual(answer, "agent answer")
            self.assertEqual(sources, agent_docs)
            self.assertFalse(route_retrieve.called)
            source_hints.assert_called_once()
            run_agentic.assert_called_once()
            self.assertEqual(run_agentic.call_args.kwargs["task_type"], "evidence")
            self.assertEqual(run_agentic.call_args.kwargs["source_hints"], ["agent-paper.pdf"])
            self.assertEqual(run_agentic.call_args.kwargs["llm_model"], "answer-model")
            self.assertEqual(run_agentic.call_args.kwargs["max_repair_rounds"], 2)
            self.assertEqual(
                generate_answer.call_args.kwargs["verified_evidence_summary"],
                "verified evidence summary",
            )

            record = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(record["route"], "agentic_mixed")
            self.assertEqual(record["retrieved_sources"][0]["file"], "agent-paper.pdf")
            self.assertEqual(record["agent_trace"], agent_trace)
            self.assertTrue(record["feature_flags"]["agentic_query"])

    def test_ask_with_context_uses_standard_route_when_agentic_disabled_for_evidence_question(self):
        docs = [Document(page_content="standard evidence", metadata={"source": "paper.pdf", "page": 2})]

        with (
            patch.dict(rag_pipeline.config, {"enable_agentic_query": False, "enable_query_logging": False}),
            patch.object(rag_pipeline, "run_agentic_rag") as run_agentic,
            patch.object(rag_pipeline, "_route_retrieve", return_value=(docs, "mixed")) as route_retrieve,
            patch.object(rag_pipeline, "_generate_answer", return_value="standard answer"),
        ):
            answer, sources = rag_pipeline.ask_with_context(
                hybrid=object(),
                conversation=FakeConversation(),
                question="这篇论文的证据在哪一页？",
            )

        self.assertEqual(answer, "standard answer")
        self.assertEqual(sources, docs)
        route_retrieve.assert_called_once()
        self.assertFalse(run_agentic.called)

    def test_ask_with_context_force_agent_overrides_disabled_config(self):
        agent_docs = [Document(page_content="forced evidence", metadata={"source": "forced.pdf", "page": 1})]

        with (
            patch.dict(rag_pipeline.config, {"enable_agentic_query": False, "enable_query_logging": False}),
            patch.object(
                rag_pipeline,
                "run_agentic_rag",
                return_value={
                    "final_docs": agent_docs,
                    "route": "agentic_mixed",
                    "agent_trace": {"enabled": True},
                    "verified_summary": "",
                },
            ) as run_agentic,
            patch.object(rag_pipeline, "_route_retrieve") as route_retrieve,
            patch.object(rag_pipeline, "_generate_answer", return_value="forced answer"),
            patch.object(rag_pipeline, "_get_llm", return_value=object()),
        ):
            answer, sources = rag_pipeline.ask_with_context(
                hybrid=object(),
                conversation=FakeConversation(),
                question="普通问题",
                force_agent=True,
            )

        self.assertEqual(answer, "forced answer")
        self.assertEqual(sources, agent_docs)
        run_agentic.assert_called_once()
        self.assertFalse(route_retrieve.called)

    def test_ask_with_context_logs_agent_trace_when_agentic_returns_no_docs(self):
        agent_trace = {"enabled": True, "fallback_reason": "no_supported_docs"}

        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "query_runs.jsonl"
            with (
                patch.dict(
                    rag_pipeline.config,
                    {
                        "enable_agentic_query": True,
                        "enable_query_logging": True,
                        "query_log_path": str(log_path),
                    },
                ),
                patch.object(
                    rag_pipeline,
                    "run_agentic_rag",
                    return_value={
                        "final_docs": None,
                        "route": "agentic_mixed",
                        "agent_trace": agent_trace,
                        "verified_summary": "",
                    },
                ),
                patch.object(rag_pipeline, "_route_retrieve") as route_retrieve,
                patch.object(rag_pipeline, "_generate_answer") as generate_answer,
                patch.object(rag_pipeline, "_get_llm", return_value=object()),
                patch.object(rag_pipeline, "_get_embedding_device", return_value="cpu"),
            ):
                answer, sources = rag_pipeline.ask_with_context(
                    hybrid=object(),
                    conversation=FakeConversation(),
                    question="这篇论文的证据在哪一页？",
                )

            self.assertEqual(answer, "❌ 未找到相关内容")
            self.assertEqual(sources, [])
            self.assertFalse(route_retrieve.called)
            self.assertFalse(generate_answer.called)
            record = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(record["route"], "agentic_mixed")
            self.assertEqual(record["retrieved_sources"], [])
            self.assertEqual(record["agent_trace"], agent_trace)

    def test_classify_agentic_task_prefers_figure_for_figure_evidence_question(self):
        self.assertEqual(
            rag_pipeline._classify_agentic_task("Figure 2 的证据在哪一页？"),
            "figure",
        )
        self.assertEqual(
            rag_pipeline._classify_agentic_task("DeepSeek-R1 的 multilingual safety performance 图在哪一页？"),
            "figure",
        )

    def test_classify_agentic_task_uses_figure_for_figure_explanation(self):
        self.assertEqual(rag_pipeline._classify_agentic_task("Figure 2 是什么？"), "figure")

    def test_classify_agentic_task_does_not_treat_image_word_as_figure(self):
        self.assertEqual(rag_pipeline._classify_agentic_task("ViT 将图像切成 patch 的证据在哪一页？"), "evidence")
        self.assertEqual(rag_pipeline._classify_agentic_task("介绍图注意力机制"), "method")


if __name__ == "__main__":
    unittest.main()
