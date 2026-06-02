import unittest

from langchain_core.documents import Document

from paper_rag.agentic.context import assemble_agentic_context, build_verified_evidence_summary


def _doc(text: str, source: str | None = "paper-a.pdf", page=None, **metadata) -> Document:
    merged = {}
    if source is not None:
        merged["source"] = source
    if page is not None:
        merged["page"] = page
    merged.update(metadata)
    return Document(page_content=text, metadata=merged)


class AgenticContextTest(unittest.TestCase):
    def test_summary_includes_statuses_claim_sources_and_answer_constraints(self):
        summary = build_verified_evidence_summary(
            [
                {
                    "goal_id": "g1",
                    "claim": "GPT-3 使用 few-shot prompting",
                    "status": "supported",
                    "supporting_sources": [{"file": "gpt3.pdf", "page": 7}],
                },
                {"goal_id": "g2", "claim": "缺失声明", "status": "unsupported", "supporting_sources": []},
            ]
        )

        self.assertTrue(summary.startswith("【已校验证据】"))
        self.assertIn("Goal g1: supported", summary)
        self.assertIn("Claim: GPT-3 使用 few-shot prompting", summary)
        self.assertIn("gpt3.pdf p7", summary)
        self.assertIn("Goal g2: unsupported", summary)
        self.assertIn("优先使用 supported 证据", summary)
        self.assertIn("partial 证据需谨慎表述", summary)
        self.assertIn("unsupported 需说明未找到足够证据", summary)
        self.assertIn("不要跨论文错归因", summary)

    def test_assemble_prioritizes_supported_docs_and_filters_noise_for_evidence(self):
        docs = [
            _doc("noise", source="bert.pdf", page=2),
            _doc("supported", source="/papers/gpt3.pdf", page=7),
            _doc("partial", source="t5.pdf", page="3"),
        ]
        verified = [
            {"goal_id": "g1", "status": "supported", "supporting_sources": [{"file": "gpt3.pdf", "page": 7}]},
            {"goal_id": "g2", "status": "partial", "supporting_sources": [{"file": "t5.pdf", "page": "3"}]},
            {"goal_id": "g3", "status": "unsupported", "supporting_sources": [{"file": "bert.pdf", "page": 2}]},
        ]

        result = assemble_agentic_context(docs, verified, task_type="evidence")

        self.assertEqual(["supported", "partial"], [doc.page_content for doc in result.final_docs])
        self.assertIn("Goal g1: supported", result.verified_summary)

    def test_figure_task_keeps_vision_docs_and_orders_them_first(self):
        docs = [
            _doc("text supported", source="paper-a.pdf", page=4),
            _doc("vision other", source="paper-b.pdf", page=10, paper_region="vision"),
            _doc("noise", source="paper-c.pdf", page=1),
        ]
        verified = [
            {"goal_id": "g1", "status": "supported", "supporting_sources": [{"file": "paper-a.pdf", "page": 4}]}
        ]

        result = assemble_agentic_context(docs, verified, task_type="figure")

        self.assertEqual(["vision other", "text supported"], [doc.page_content for doc in result.final_docs])

    def test_source_file_windows_path_and_page_string_key_can_match(self):
        docs = [
            _doc("match", source=None, source_file=r"C:\papers\gpt3.pdf", page="7"),
            _doc("noise", source=None, source_file=r"C:\papers\bert.pdf", page="7"),
        ]
        verified = [
            {"goal_id": "g1", "status": "supported", "supporting_sources": [{"file": "/tmp/gpt3.pdf", "page": "7"}]}
        ]

        result = assemble_agentic_context(docs, verified, task_type="followup")

        self.assertEqual(["match"], [doc.page_content for doc in result.final_docs])

    def test_evidence_and_followup_filter_unrelated_vision_docs_when_supported_keys_exist(self):
        docs = [
            _doc("supported", source="gpt3.pdf", page=7),
            _doc("vision noise", source="bert.pdf", page=2, paper_region="vision"),
            _doc("text noise", source="t5.pdf", page=3),
        ]
        verified = [
            {"goal_id": "g1", "status": "supported", "supporting_sources": [{"file": "gpt3.pdf", "page": 7}]}
        ]

        for task_type in ["evidence", "followup"]:
            with self.subTest(task_type=task_type):
                result = assemble_agentic_context(docs, verified, task_type=task_type)

                self.assertEqual(["supported"], [doc.page_content for doc in result.final_docs])


if __name__ == "__main__":
    unittest.main()
