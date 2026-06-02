import contextlib
import io
import unittest

from langchain_core.documents import Document

from paper_rag.agentic.verifier import verify_goal
from utils.prompt_loader import load_prompt


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


def _doc(text: str, source: str = "paper-a.pdf", page: int = 3) -> Document:
    return Document(page_content=text, metadata={"source": source, "page": page})


class AgenticVerifierTest(unittest.TestCase):
    def test_keyword_fallback_supported_uses_doc_sources(self):
        result = verify_goal(
            {"id": "g1", "claim": "contrastive learning improves retrieval"},
            [_doc("This section shows contrastive learning improves dense retrieval.")],
        )

        self.assertEqual("g1", result["goal_id"])
        self.assertEqual("contrastive learning improves retrieval", result["claim"])
        self.assertEqual("supported", result["status"])
        self.assertEqual("paper-a.pdf", result["supporting_sources"][0]["file"])
        self.assertEqual(3, result["supporting_sources"][0]["page"])

    def test_keyword_fallback_partial_and_unsupported_boundaries(self):
        partial = verify_goal(
            {"id": "g2", "claim": "alpha beta gamma delta"},
            [_doc("Only alpha appears in this evidence.")],
        )
        unsupported = verify_goal(
            {"id": "g3", "claim": "alpha beta"},
            [_doc("No matching vocabulary here.")],
        )

        self.assertEqual("partial", partial["status"])
        self.assertTrue(partial["supporting_sources"])
        self.assertEqual("unsupported", unsupported["status"])
        self.assertEqual([], unsupported["supporting_sources"])

    def test_empty_docs_is_unsupported_without_sources(self):
        result = verify_goal({"id": "empty", "claim": "alpha beta"}, [])

        self.assertEqual("unsupported", result["status"])
        self.assertEqual([], result["supporting_sources"])

    def test_short_english_abbreviations_do_not_substring_match(self):
        result = verify_goal(
            {"id": "g-short", "claim": "AI QA"},
            [_doc("This paper reports fair evaluation results.")],
        )

        self.assertEqual("unsupported", result["status"])
        self.assertEqual([], result["supporting_sources"])

    def test_english_abbreviations_match_case_insensitive_tokens(self):
        result = verify_goal(
            {"id": "g-abbrev", "claim": "AI QA"},
            [_doc("AI improves QA across the benchmark.")],
        )

        self.assertEqual("supported", result["status"])
        self.assertTrue(result["supporting_sources"])

    def test_partial_when_only_one_of_two_terms_matches(self):
        result = verify_goal(
            {"id": "g-one-hit", "claim": "AI QA"},
            [_doc("AI improves systems.")],
        )

        self.assertEqual("partial", result["status"])
        self.assertTrue(result["supporting_sources"])

    def test_hyphenated_english_terms_match_as_tokens(self):
        result = verify_goal(
            {"id": "g-hyphen", "claim": "self-attention retrieval"},
            [_doc("Self-attention improves retrieval quality.")],
        )

        self.assertEqual("supported", result["status"])

    def test_chinese_long_claim_matches_split_evidence(self):
        result = verify_goal(
            {"id": "g-cjk", "claim": "检索增强生成能减少幻觉"},
            [_doc("检索增强方法有助于减少模型幻觉。")],
        )

        self.assertIn(result["status"], {"partial", "supported"})
        self.assertTrue(result["supporting_sources"])

    def test_llm_valid_json_can_override_fallback_and_clean_missing_terms(self):
        llm = FakeLLM('{"status": "partial", "reason": "some support", "missing_terms": ["delta", null, "  ", 42]}')

        result = verify_goal(
            {"id": "g4", "claim": "alpha beta gamma"},
            [_doc("alpha beta gamma are all present.")],
            llm=llm,
        )

        self.assertEqual("partial", result["status"])
        self.assertEqual(["delta", "42"], result["missing_terms"])
        self.assertTrue(result["supporting_sources"])

    def test_llm_invalid_status_keeps_keyword_fallback(self):
        llm = FakeLLM('{"status": "certain", "missing_terms": ["beta"]}')

        result = verify_goal(
            {"id": "g5", "claim": "alpha beta"},
            [_doc("alpha beta are both present.")],
            llm=llm,
        )

        self.assertEqual("supported", result["status"])
        self.assertEqual(["beta"], result["missing_terms"])

    def test_broken_llm_and_dirty_json_fallback_without_raising(self):
        broken = verify_goal(
            {"id": "g6", "claim": "alpha beta gamma"},
            [_doc("alpha appears once.")],
            llm=BrokenLLM(),
        )
        dirty = verify_goal(
            {"id": "g7", "claim": "alpha beta"},
            [_doc("alpha beta both appear.")],
            llm=FakeLLM("not json at all"),
        )

        self.assertEqual("partial", broken["status"])
        self.assertEqual("supported", dirty["status"])

    def test_prompt_formats_and_contains_claim_query_evidence(self):
        with contextlib.redirect_stdout(io.StringIO()):
            template = load_prompt("agent_verifier_prompt")
        formatted = template.format(claim="声明A", query="查询B", evidence="证据C")
        self.assertIn("声明A", formatted)
        self.assertIn("查询B", formatted)
        self.assertIn("证据C", formatted)

        llm = FakeLLM('{"status": "supported", "missing_terms": []}')
        verify_goal(
            {"id": "g8", "claim": "声明A", "query": "查询B"},
            [_doc("证据C", source="paper-c.pdf", page=8)],
            llm=llm,
        )

        self.assertIn("声明A", llm.prompts[0])
        self.assertIn("查询B", llm.prompts[0])
        self.assertIn("证据C", llm.prompts[0])
        self.assertIn("paper-c.pdf", llm.prompts[0])
        self.assertIn("page=8", llm.prompts[0])


if __name__ == "__main__":
    unittest.main()
