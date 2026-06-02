import unittest

from langchain_core.documents import Document

from paper_rag.agentic.graph import run_agentic_rag


def _doc(text: str, source: str = "paper-a.pdf", page: int = 1) -> Document:
    return Document(page_content=text, metadata={"source": source, "page": page})


class FakeRouter:
    def __init__(self, docs_by_query=None, default_docs=None):
        self.docs_by_query = docs_by_query or {}
        self.default_docs = default_docs or []
        self.calls = []

    def route(self, hybrid, query, llm_model="", temperature=0.0):
        self.calls.append(
            {
                "hybrid": hybrid,
                "query": query,
                "llm_model": llm_model,
                "temperature": temperature,
            }
        )
        return list(self.docs_by_query.get(query, self.default_docs)), "base_route"


class AgenticGraphTest(unittest.TestCase):
    def test_graph_runs_plan_collect_verify_and_assemble(self):
        router = FakeRouter(
            default_docs=[
                _doc("attention mechanisms support evidence retrieval", source="attention.pdf", page=3),
            ]
        )

        result = run_agentic_rag(
            question="How does attention support evidence retrieval?",
            standalone_question="attention mechanisms support evidence retrieval",
            task_type="evidence",
            source_hints=["attention.pdf"],
            hybrid=object(),
            router=router,
            llm_model="qwen",
            temperature=0.2,
        )

        self.assertEqual("agentic_page_evidence", result["route"])
        self.assertEqual(["attention mechanisms support evidence retrieval"], [call["query"] for call in router.calls])
        self.assertEqual("qwen", router.calls[0]["llm_model"])
        self.assertEqual(0.2, router.calls[0]["temperature"])
        self.assertEqual(["attention mechanisms support evidence retrieval"], [goal["query"] for goal in result["goals"]])
        self.assertEqual(["attention mechanisms support evidence retrieval"], [doc.page_content for doc in result["final_docs"]])
        self.assertTrue(result["agent_trace"]["enabled"])
        self.assertIn("Goal", result["verified_summary"])

    def test_repair_stops_after_one_round_when_evidence_is_missing(self):
        router = FakeRouter()

        result = run_agentic_rag(
            question="Missing claim?",
            standalone_question="missing claim",
            task_type="evidence",
            source_hints=[],
            hybrid=object(),
            router=router,
        )

        self.assertEqual(1, result["agent_trace"]["repair_rounds"])
        self.assertEqual(["g1"], result["missing_goal_ids"])
        self.assertEqual("unsupported", result["verified_evidence"][0]["status"])
        self.assertFalse(result["agent_trace"]["repair_success"])
        self.assertEqual([], result["final_docs"])

    def test_compare_task_collects_docs_for_multiple_goals(self):
        docs_by_query = {
            "compare transformers paper-a.pdf": [_doc("compare transformers paper-a", "paper-a.pdf", 2)],
            "compare transformers paper-b.pdf": [_doc("compare transformers paper-b", "paper-b.pdf", 4)],
        }
        router = FakeRouter(docs_by_query=docs_by_query)

        result = run_agentic_rag(
            question="Compare transformers",
            standalone_question="compare transformers",
            task_type="compare",
            source_hints=["paper-a.pdf", "paper-b.pdf"],
            hybrid=object(),
            router=router,
        )

        self.assertEqual("agentic_compare", result["route"])
        self.assertEqual(2, len(result["goals"]))
        self.assertEqual(
            ["compare transformers paper-a", "compare transformers paper-b"],
            [doc.page_content for doc in result["collected_docs"]],
        )
        self.assertEqual(
            ["paper-a.pdf", "paper-b.pdf"],
            [doc.metadata["source"] for doc in result["final_docs"]],
        )
        self.assertEqual(
            [["paper-a.pdf"], ["paper-b.pdf"]],
            [
                [source["file"] for source in evidence["supporting_sources"]]
                for evidence in result["verified_evidence"]
            ],
        )

    def test_max_repair_rounds_zero_does_not_repair(self):
        router = FakeRouter()

        result = run_agentic_rag(
            question="Missing claim?",
            standalone_question="missing claim",
            task_type="evidence",
            source_hints=[],
            hybrid=object(),
            router=router,
            max_repair_rounds=0,
        )

        self.assertEqual(0, result["agent_trace"]["repair_rounds"])
        self.assertEqual(["g1"], result["missing_goal_ids"])
        self.assertEqual("unsupported", result["verified_evidence"][0]["status"])


if __name__ == "__main__":
    unittest.main()
