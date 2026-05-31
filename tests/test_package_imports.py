import unittest


class PackageImportsTest(unittest.TestCase):
    def test_public_package_reexports_current_services(self):
        from paper_rag.config import RagSettings
        from paper_rag.generation import build_rag_prompt, prepare_docs_for_context
        from paper_rag.observability import TraceTimer, build_query_log_record, source_from_doc, write_query_log
        from paper_rag.retrieval import RetrievalRouter, is_overview_question
        from paper_rag.ui import build_feedback_payload, clear_conversation_state

        self.assertTrue(callable(RagSettings.from_mapping))
        self.assertTrue(callable(build_rag_prompt))
        self.assertTrue(callable(prepare_docs_for_context))
        self.assertTrue(callable(build_query_log_record))
        self.assertTrue(callable(source_from_doc))
        self.assertTrue(callable(TraceTimer))
        self.assertTrue(callable(write_query_log))
        self.assertTrue(callable(RetrievalRouter))
        self.assertTrue(is_overview_question("什么是 Transformer？"))
        self.assertTrue(callable(build_feedback_payload))
        self.assertTrue(callable(clear_conversation_state))


if __name__ == "__main__":
    unittest.main()
