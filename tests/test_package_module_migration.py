import unittest
from unittest.mock import patch


class PackageModuleMigrationTest(unittest.TestCase):
    def test_low_coupling_implementations_live_under_package_modules(self):
        from paper_rag.generation.service import build_rag_prompt
        from paper_rag.observability.sources import source_from_doc
        from paper_rag.ui.state import clear_conversation_state
        from paper_rag.ui.services import build_feedback_payload

        self.assertTrue(callable(build_rag_prompt))
        self.assertTrue(callable(source_from_doc))
        self.assertTrue(callable(clear_conversation_state))
        self.assertTrue(callable(build_feedback_payload))

    def test_medium_coupling_implementations_live_under_package_modules(self):
        from paper_rag.generation.context import prepare_docs_for_context
        from paper_rag.retrieval.router import RetrievalRouter

        self.assertTrue(callable(prepare_docs_for_context))
        self.assertTrue(callable(RetrievalRouter))

    def test_remaining_helper_implementations_live_under_package_modules(self):
        from paper_rag.generation.parent_retrieval import expand_parent_pages
        from paper_rag.observability.feedback import build_feedback_record
        from paper_rag.observability.query_logger import build_query_log_record
        from paper_rag.observability.service import write_query_log
        from paper_rag.observability.trace import TraceTimer

        self.assertTrue(callable(expand_parent_pages))
        self.assertTrue(callable(build_feedback_record))
        self.assertTrue(callable(build_query_log_record))
        self.assertTrue(callable(write_query_log))
        self.assertTrue(callable(TraceTimer))

    def test_remaining_retrieval_and_context_implementations_have_package_paths(self):
        from paper_rag.generation.context_compression import compress_chunk, compress_documents
        from paper_rag.retrieval.hybrid import HybridRetriever, SemanticWeightDecider
        from paper_rag.retrieval.query_expansion import expand_query, filter_query_variants
        from paper_rag.retrieval.reranker import Reranker, apply_rerank

        self.assertTrue(callable(compress_chunk))
        self.assertTrue(callable(compress_documents))
        self.assertTrue(callable(HybridRetriever))
        self.assertTrue(callable(SemanticWeightDecider))
        self.assertTrue(callable(expand_query))
        self.assertTrue(callable(filter_query_variants))
        self.assertTrue(callable(Reranker))
        self.assertTrue(callable(apply_rerank))

    def test_package_modules_do_not_depend_on_root_compat_wrappers(self):
        from pathlib import Path

        package_files = [
            Path("paper_rag/generation/context.py"),
            Path("paper_rag/retrieval/router.py"),
            Path("paper_rag/ui/__init__.py"),
            Path("paper_rag/ui/services.py"),
            Path("rag_pipeline.py"),
            Path("build_knowledge.py"),
        ]
        forbidden = [
            "from app_state import",
            "from feedback import",
            "from context_compression import",
            "from hybrid_retriever import",
            "from parent_retrieval import",
            "from query_expansion import",
            "from reranker import",
        ]

        for path in package_files:
            text = path.read_text(encoding="utf-8")
            for pattern in forbidden:
                self.assertNotIn(pattern, text, f"{path} still imports root wrapper via {pattern}")

    def test_root_compat_wrappers_document_package_replacement(self):
        import importlib

        wrappers = [
            "context_builder",
            "context_compression",
            "app_state",
            "app_services",
            "feedback",
            "generation_service",
            "hybrid_retriever",
            "parent_retrieval",
            "query_expansion",
            "query_logger",
            "reranker",
            "retrieval_router",
            "source_utils",
        ]

        for module_name in wrappers:
            module = importlib.import_module(module_name)
            doc = module.__doc__ or ""
            self.assertIn("兼容薄壳", doc, f"{module_name} should document its compatibility role")
            self.assertIn("paper_rag.", doc, f"{module_name} should point callers to package imports")

    def test_rag_pipeline_single_turn_entries_keep_patch_compatibility(self):
        import rag_pipeline

        with (
            patch.object(rag_pipeline, "_route_retrieve", return_value=(["doc"], "mixed")) as route,
            patch.object(rag_pipeline, "_generate_answer", return_value="answer") as generate,
        ):
            answer, docs = rag_pipeline.route_question("hybrid", "question", llm_model="model", temperature=0.1)

        self.assertEqual((answer, docs), ("answer", ["doc"]))
        route.assert_called_once_with("hybrid", "question", "model", 0.1)
        generate.assert_called_once_with(
            "question",
            ["doc"],
            llm_model="model",
            temperature=0.1,
            hybrid="hybrid",
        )

        with (
            patch.object(rag_pipeline, "_retrieve_with_hyde", return_value=["hyde-doc"]) as retrieve,
            patch.object(rag_pipeline, "_generate_answer", return_value="hyde answer") as generate,
        ):
            answer, docs = rag_pipeline.ask_with_hyde("hybrid", "question", llm_model="model", temperature=0.2)

        self.assertEqual((answer, docs), ("hyde answer", ["hyde-doc"]))
        retrieve.assert_called_once_with("hybrid", "question", "model", 0.2)
        generate.assert_called_once_with(
            "question",
            ["hyde-doc"],
            llm_model="model",
            temperature=0.2,
            hybrid="hybrid",
        )


if __name__ == "__main__":
    unittest.main()
