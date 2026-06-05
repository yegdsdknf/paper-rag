import unittest


class FakeConversation:
    def __init__(self, history=None, rewritten="standalone question"):
        self.history = history or []
        self.rewritten = rewritten
        self.reformulate_calls = []

    def reformulate(self, question):
        self.reformulate_calls.append(question)
        return self.rewritten


class PipelineServiceTest(unittest.TestCase):
    def test_reformulate_question_can_force_rewrite_without_history(self):
        from paper_rag.pipeline.service import reformulate_question

        conversation = FakeConversation(history=[], rewritten="standalone question")

        result = reformulate_question(conversation, "follow up", require_history=False)

        self.assertEqual(result.standalone_question, "standalone question")
        self.assertTrue(result.rewritten)
        self.assertEqual(conversation.reformulate_calls, ["follow up"])

    def test_reformulate_question_skips_rewrite_when_history_required_but_empty(self):
        from paper_rag.pipeline.service import reformulate_question

        conversation = FakeConversation(history=[], rewritten="standalone question")

        result = reformulate_question(conversation, "first question", require_history=True)

        self.assertEqual(result.standalone_question, "first question")
        self.assertFalse(result.rewritten)
        self.assertEqual(conversation.reformulate_calls, [])

    def test_reformulate_question_reports_unchanged_question_as_not_rewritten(self):
        from paper_rag.pipeline.service import reformulate_question

        conversation = FakeConversation(history=[{"role": "user", "content": "history"}], rewritten="same question")

        result = reformulate_question(conversation, "same question", require_history=True)

        self.assertEqual(result.standalone_question, "same question")
        self.assertFalse(result.rewritten)
        self.assertEqual(conversation.reformulate_calls, ["same question"])

    def test_write_pipeline_query_log_includes_trace_for_mixed_route(self):
        from paper_rag.pipeline.service import write_pipeline_query_log

        calls = []

        write_pipeline_query_log(
            settings="settings",
            question="original",
            standalone_question="standalone",
            route="mixed:expanded",
            llm_model="qwen2.5:3b",
            docs=["doc"],
            elapsed={"total": 1.0},
            embedding_device_fn="device-fn",
            query_trace={
                "variants": ["variant"],
                "rejections": [{"variant": "bad", "reason": "too_distant"}],
            },
            context_stats={"input_chars": 10},
            write_query_log_fn=lambda **kwargs: calls.append(kwargs),
        )

        self.assertEqual(calls[0]["settings"], "settings")
        self.assertEqual(calls[0]["query_variants"], ["variant"])
        self.assertEqual(calls[0]["query_variant_rejections"], [{"variant": "bad", "reason": "too_distant"}])
        self.assertEqual(calls[0]["context_stats"], {"input_chars": 10})

    def test_write_pipeline_query_log_clears_trace_for_non_mixed_route(self):
        from paper_rag.pipeline.service import write_pipeline_query_log

        calls = []

        write_pipeline_query_log(
            settings="settings",
            question="original",
            standalone_question="standalone",
            route="hyde",
            llm_model="qwen2.5:3b",
            docs=[],
            elapsed={"total": 1.0},
            embedding_device_fn="device-fn",
            query_trace={
                "variants": ["variant"],
                "rejections": [{"variant": "bad", "reason": "too_distant"}],
            },
            error="LLM unavailable",
            write_query_log_fn=lambda **kwargs: calls.append(kwargs),
        )

        self.assertEqual(calls[0]["query_variants"], [])
        self.assertEqual(calls[0]["query_variant_rejections"], [])
        self.assertEqual(calls[0]["error"], "LLM unavailable")

    def test_handle_no_docs_response_writes_log_and_returns_answer_tuple(self):
        from paper_rag.pipeline.service import handle_no_docs_response

        log_calls = []

        result = handle_no_docs_response(
            question="original",
            standalone_question="standalone",
            route="hyde",
            llm_model="qwen2.5:3b",
            elapsed={"total": 0.3},
            write_query_log_fn=lambda **kwargs: log_calls.append(kwargs),
        )

        self.assertEqual(result, ("❌ 未找到相关内容", []))
        self.assertEqual(log_calls[0]["docs"], [])
        self.assertEqual(log_calls[0]["route"], "hyde")

    def test_handle_no_docs_response_can_return_stream_events(self):
        from paper_rag.pipeline.service import handle_no_docs_response

        log_calls = []

        events = handle_no_docs_response(
            question="original",
            standalone_question="standalone",
            route="mixed",
            llm_model="qwen2.5:3b",
            elapsed={"total": 0.3},
            stream=True,
            write_query_log_fn=lambda **kwargs: log_calls.append(kwargs),
        )

        self.assertEqual(events, [{"type": "token", "data": "❌ 未找到相关内容"}])
        self.assertEqual(log_calls[0]["standalone_question"], "standalone")
        self.assertEqual(log_calls[0]["elapsed"], {"total": 0.3})


if __name__ == "__main__":
    unittest.main()
