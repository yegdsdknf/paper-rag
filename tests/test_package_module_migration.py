import unittest


class PackageModuleMigrationTest(unittest.TestCase):
    def test_low_coupling_implementations_live_under_package_modules(self):
        from paper_rag.generation.service import build_rag_prompt
        from paper_rag.observability.sources import source_from_doc
        from paper_rag.ui.services import build_feedback_payload

        self.assertTrue(callable(build_rag_prompt))
        self.assertTrue(callable(source_from_doc))
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
        from paper_rag.retrieval.query_expansion import expand_query, filter_query_variants
        from paper_rag.retrieval.reranker import Reranker, apply_rerank

        self.assertTrue(callable(compress_chunk))
        self.assertTrue(callable(compress_documents))
        self.assertTrue(callable(expand_query))
        self.assertTrue(callable(filter_query_variants))
        self.assertTrue(callable(Reranker))
        self.assertTrue(callable(apply_rerank))

    def test_package_modules_do_not_depend_on_root_compat_wrappers(self):
        from pathlib import Path

        package_files = [
            Path("paper_rag/generation/context.py"),
            Path("paper_rag/retrieval/router.py"),
        ]
        forbidden = [
            "from context_compression import",
            "from parent_retrieval import",
            "from query_expansion import",
            "from reranker import",
        ]

        for path in package_files:
            text = path.read_text(encoding="utf-8")
            for pattern in forbidden:
                self.assertNotIn(pattern, text, f"{path} still imports root wrapper via {pattern}")


if __name__ == "__main__":
    unittest.main()
