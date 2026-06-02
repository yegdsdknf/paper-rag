import unittest

from langchain_core.documents import Document

from paper_rag.agentic.collector import collect_for_goal


def _doc(text: str, source: str = "paper-a.pdf", page: int = 1, **metadata) -> Document:
    merged = {"source": source, "page": page}
    merged.update(metadata)
    return Document(page_content=text, metadata=merged)


class FakeRouter:
    def __init__(self, docs):
        self.docs = docs
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
        return list(self.docs), "original_route"


class FakeVectorStore:
    def get(self, include=None):
        self.include = include
        return {
            "documents": ["vision a", "text b", "vision c"],
            "metadatas": [
                {"source": "paper-a.pdf", "page": "4", "paper_region": "vision"},
                {"source": "paper-a.pdf", "page": 5},
                {"source_file": r"C:\papers\paper-b.pdf", "page": 6, "chunk_strategy": "vision_summary"},
            ],
        }


class FakeHybrid:
    vector_store = FakeVectorStore()


class BrokenVectorStore:
    def get(self, include=None):
        raise RuntimeError("vector store unavailable")


class BrokenHybrid:
    vector_store = BrokenVectorStore()


class DocumentVectorStore:
    def __init__(self, doc):
        self.doc = doc

    def get(self, include=None):
        return {
            "documents": [self.doc],
            "metadatas": [{"source": "sidecar.pdf", "page": 9, "paper_region": "vision"}],
        }


class DocumentHybrid:
    def __init__(self, doc):
        self.vector_store = DocumentVectorStore(doc)


class AgenticCollectorTest(unittest.TestCase):
    def test_page_evidence_uses_goal_query_route_and_source_hint_filter(self):
        docs = [
            _doc("matched", source=r"C:\papers\gpt3.pdf", page=7),
            _doc("noise", source="bert.pdf", page=2),
        ]
        router = FakeRouter(docs)

        collected, route = collect_for_goal(
            {"goal_type": "page_evidence", "claim": "claim text", "query": "specific query", "source_hint": "gpt3.pdf"},
            hybrid=object(),
            router=router,
            llm_model="qwen",
            temperature=0.2,
        )

        self.assertEqual("agentic_page_evidence", route)
        self.assertEqual(["specific query"], [call["query"] for call in router.calls])
        self.assertEqual("qwen", router.calls[0]["llm_model"])
        self.assertEqual(0.2, router.calls[0]["temperature"])
        self.assertEqual(["matched"], [doc.page_content for doc in collected])

    def test_page_evidence_falls_back_to_claim_query_when_query_missing(self):
        router = FakeRouter([_doc("result")])

        collect_for_goal({"goal_type": "page_evidence", "claim": "claim only"}, hybrid=object(), router=router)

        self.assertEqual("claim only", router.calls[0]["query"])

    def test_compare_and_method_route_names(self):
        for goal_type, expected_route in [
            ("compare_dimension", "agentic_compare"),
            ("method_overview", "agentic_method"),
        ]:
            with self.subTest(goal_type=goal_type):
                router = FakeRouter([_doc("result")])
                docs, route = collect_for_goal(
                    {"goal_type": goal_type, "claim": "claim", "query": "query"},
                    hybrid=object(),
                    router=router,
                )

                self.assertEqual(expected_route, route)
                self.assertEqual(["result"], [doc.page_content for doc in docs])

    def test_figure_prefers_vision_loader_docs(self):
        router = FakeRouter([_doc("router result")])

        docs, route = collect_for_goal(
            {"goal_type": "figure_evidence", "claim": "figure", "source_hint": "paper-a.pdf"},
            hybrid=object(),
            router=router,
            vision_loader=lambda hybrid, source_hint: [_doc("vision result", source="paper-a.pdf", paper_region="vision")],
        )

        self.assertEqual("agentic_figure", route)
        self.assertEqual(["vision result"], [doc.page_content for doc in docs])
        self.assertEqual([], router.calls)

    def test_figure_falls_back_to_router_when_no_vision_docs(self):
        router = FakeRouter([_doc("router result")])

        docs, route = collect_for_goal(
            {"goal_type": "figure_evidence", "claim": "figure claim"},
            hybrid=object(),
            router=router,
            vision_loader=lambda hybrid, source_hint: [],
        )

        self.assertEqual("agentic_figure_text_fallback", route)
        self.assertEqual(["router result"], [doc.page_content for doc in docs])
        self.assertEqual("figure claim", router.calls[0]["query"])

    def test_default_vision_loader_reads_vector_store_and_filters_by_source_hint(self):
        router = FakeRouter([])

        docs, route = collect_for_goal(
            {"goal_type": "figure_evidence", "claim": "figure", "source_hint": "paper-b.pdf"},
            hybrid=FakeHybrid(),
            router=router,
        )

        self.assertEqual("agentic_figure", route)
        self.assertEqual(["vision c"], [doc.page_content for doc in docs])
        self.assertEqual("paper-b.pdf", docs[0].metadata["source_file"].replace("\\", "/").split("/")[-1])

    def test_default_vision_loader_error_falls_back_to_router(self):
        router = FakeRouter([_doc("router fallback")])

        docs, route = collect_for_goal(
            {"goal_type": "figure_evidence", "claim": "figure claim"},
            hybrid=BrokenHybrid(),
            router=router,
        )

        self.assertEqual("agentic_figure_text_fallback", route)
        self.assertEqual(["router fallback"], [doc.page_content for doc in docs])
        self.assertEqual("figure claim", router.calls[0]["query"])

    def test_default_vision_loader_copies_document_metadata_without_mutating_original(self):
        original = _doc("vision document", source="content.pdf", page=3, paper_region="vision", content_only=True)
        router = FakeRouter([])

        docs, route = collect_for_goal(
            {"goal_type": "figure_evidence", "claim": "figure"},
            hybrid=DocumentHybrid(original),
            router=router,
        )

        self.assertEqual("agentic_figure", route)
        self.assertIsNot(original, docs[0])
        self.assertEqual("vision document", docs[0].page_content)
        self.assertEqual("sidecar.pdf", docs[0].metadata["source"])
        self.assertEqual(9, docs[0].metadata["page"])
        self.assertEqual("content.pdf", original.metadata["source"])
        self.assertEqual(3, original.metadata["page"])


if __name__ == "__main__":
    unittest.main()
