import unittest

from langchain_core.documents import Document

from paper_rag.agentic.json_utils import parse_json_object
from paper_rag.agentic.schema import (
    AgenticRagState,
    VerifiedEvidence,
    docs_to_agent_sources,
    normalize_goal,
    normalize_verified_evidence,
)


class AgenticSchemaTest(unittest.TestCase):
    def test_parse_json_object_extracts_json_from_noisy_model_output(self):
        text = "结果如下：\n```json\n{\"goals\": [{\"id\": \"g1\"}]}\n```"

        parsed = parse_json_object(text)

        self.assertEqual(parsed["goals"][0]["id"], "g1")

    def test_parse_json_object_skips_non_object_candidates(self):
        text = "```json\n[1, 2]\n```\n最终：{\"goals\": [{\"id\": \"g2\"}]}"

        parsed = parse_json_object(text)

        self.assertEqual(parsed["goals"][0]["id"], "g2")

    def test_parse_json_object_skips_braced_noise_before_object(self):
        text = "请替换 {id} 占位符，然后输出 {\"goals\": [{\"id\": \"g3\"}]}"

        parsed = parse_json_object(text)

        self.assertEqual(parsed["goals"][0]["id"], "g3")

    def test_normalize_goal_limits_status_and_source_hint(self):
        goal = normalize_goal(
            {
                "id": "",
                "goal_type": "weird",
                "claim": "GPT-3 uses Transformer",
                "query": "",
                "source_hint": "not-in-index.pdf",
                "page_hint": "7",
            },
            index=0,
            allowed_sources={"gpt3.pdf"},
        )

        self.assertEqual(goal["id"], "g1")
        self.assertEqual(goal["goal_type"], "method_overview")
        self.assertEqual(goal["query"], "GPT-3 uses Transformer")
        self.assertEqual(goal["source_hint"], "")
        self.assertEqual(goal["page_hint"], 7)

    def test_docs_to_agent_sources_serializes_documents(self):
        docs = [
            Document(
                page_content="Figure 14 | Multilingual safety performance",
                metadata={"source": "./papers/deepseekr1.pdf", "page": 51},
            )
        ]

        sources = docs_to_agent_sources(docs)

        self.assertEqual(sources[0]["file"], "deepseekr1.pdf")
        self.assertEqual(sources[0]["page"], 51)
        self.assertIn("Figure 14", sources[0]["content_preview"])

    def test_normalize_goal_strips_model_output_fields(self):
        goal = normalize_goal(
            {
                "id": "   ",
                "goal_type": " page_evidence ",
                "claim": "  GPT-3 uses Transformer  ",
                "query": "  transformer evidence  ",
                "source_hint": " ./papers/gpt3.pdf ",
            },
            index=1,
            allowed_sources={"gpt3.pdf"},
        )

        self.assertEqual(goal["id"], "g2")
        self.assertEqual(goal["goal_type"], "page_evidence")
        self.assertEqual(goal["claim"], "GPT-3 uses Transformer")
        self.assertEqual(goal["query"], "transformer evidence")
        self.assertEqual(goal["source_hint"], "gpt3.pdf")

    def test_docs_to_agent_sources_uses_source_file_fallback(self):
        docs = [
            Document(
                page_content="Appendix evidence",
                metadata={"source_file": "C:\\papers\\gpt3.pdf", "page": 3},
            )
        ]

        sources = docs_to_agent_sources(docs)

        self.assertEqual(sources[0]["file"], "gpt3.pdf")

    def test_docs_to_agent_sources_uses_unknown_without_source(self):
        sources = docs_to_agent_sources([Document(page_content="evidence", metadata={"page": 3})])

        self.assertEqual(sources[0]["file"], "unknown")

    def test_normalize_verified_evidence_cleans_status_and_fields(self):
        valid = normalize_verified_evidence(
            {
                "goal_id": " raw-goal ",
                "claim": "  GPT-3 is autoregressive  ",
                "status": " partial ",
                "supporting_sources": [{"file": "gpt3.pdf"}],
                "missing_terms": [" data ", "", 7],
            },
            goal_id=" param-goal ",
        )
        invalid = normalize_verified_evidence({"status": "unknown", "missing_terms": "data"})
        blank = normalize_verified_evidence({"status": "   "})

        self.assertEqual(valid["goal_id"], "param-goal")
        self.assertEqual(valid["claim"], "GPT-3 is autoregressive")
        self.assertEqual(valid["status"], "partial")
        self.assertEqual(valid["supporting_sources"], [{"file": "gpt3.pdf"}])
        self.assertEqual(valid["missing_terms"], ["data", "7"])
        self.assertEqual(invalid["status"], "unsupported")
        self.assertEqual(invalid["missing_terms"], [])
        self.assertEqual(blank["status"], "unsupported")

    def test_normalize_verified_evidence_filters_source_and_term_values(self):
        evidence = normalize_verified_evidence(
            {
                "supporting_sources": [
                    {"file": "gpt3.pdf"},
                    "bad",
                    7,
                    ["also bad"],
                    {"page": 3},
                ],
                "missing_terms": [None, " data ", "", 0],
            }
        )

        self.assertEqual(evidence["supporting_sources"], [{"file": "gpt3.pdf"}, {"page": 3}])
        self.assertEqual(evidence["missing_terms"], ["data", "0"])

    def test_agentic_typed_dicts_include_planned_fields(self):
        verified_fields = {
            "goal_id",
            "claim",
            "status",
            "supporting_sources",
            "missing_terms",
        }
        state_fields = {
            "question",
            "standalone_question",
            "task_type",
            "route",
            "source_hints",
            "goals",
            "collected_docs",
            "verified_evidence",
            "final_docs",
            "repair_rounds",
            "missing_goal_ids",
            "fallback_reason",
            "agent_trace",
            "answer",
            "sources",
            "elapsed",
        }

        self.assertTrue(verified_fields.issubset(VerifiedEvidence.__annotations__))
        self.assertTrue(state_fields.issubset(AgenticRagState.__annotations__))
