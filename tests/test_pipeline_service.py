import unittest


class FakeConversation:
    def __init__(self, history=None, rewritten="standalone question"):
        self.history = history or []
        self.rewritten = rewritten
        self.reformulate_calls = []
        self.format_history_calls = 0

    def reformulate(self, question):
        self.reformulate_calls.append(question)
        return self.rewritten

    def format_history(self):
        self.format_history_calls += 1
        return "formatted history"


class FakeTimer:
    def __init__(self):
        self.durations = [0.1, 0.2, 0.3]
        self.started = []

    def start_stage(self):
        marker = f"stage-{len(self.started)}"
        self.started.append(marker)
        return marker

    def elapsed_since(self, _start):
        return self.durations.pop(0)

    def elapsed_map(self, rewrite, retrieve, generate):
        return {
            "rewrite": rewrite,
            "retrieve": retrieve,
            "generate": generate,
            "total": round(rewrite + retrieve + generate, 3),
        }


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

    def test_format_conversation_history_delegates_to_conversation(self):
        from paper_rag.pipeline.service import format_conversation_history

        conversation = FakeConversation()

        self.assertEqual(format_conversation_history(conversation), "formatted history")
        self.assertEqual(conversation.format_history_calls, 1)

    def test_write_rewrite_notice_prints_only_when_rewritten(self):
        from paper_rag.pipeline.service import ReformulationResult, write_rewrite_notice

        messages = []

        write_rewrite_notice(ReformulationResult("standalone question", rewritten=True), print_fn=messages.append)
        write_rewrite_notice(ReformulationResult("original question", rewritten=False), print_fn=messages.append)

        self.assertEqual(messages, ['🔄 改写追问: "standalone question"'])

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

    def test_build_no_docs_response_returns_message_and_empty_docs(self):
        from paper_rag.pipeline.service import build_no_docs_response

        self.assertEqual(build_no_docs_response(), ("❌ 未找到相关内容", []))
        self.assertEqual(build_no_docs_response("custom empty"), ("custom empty", []))

    def test_route_question_delegates_routed_docs_to_answer_generation(self):
        from paper_rag.pipeline.service import route_question

        calls = []

        def route_retrieve(hybrid, question, llm_model, temperature):
            calls.append(("route", hybrid, question, llm_model, temperature))
            return ["doc"], "mixed"

        def generate_answer(question, docs, *, llm_model, temperature, hybrid):
            calls.append(("generate", question, docs, llm_model, temperature, hybrid))
            return "answer"

        result = route_question(
            hybrid="hybrid",
            question="question",
            llm_model="qwen2.5:3b",
            temperature=0.1,
            route_retrieve_fn=route_retrieve,
            generate_answer_fn=generate_answer,
        )

        self.assertEqual(result, ("answer", ["doc"]))
        self.assertEqual(
            calls,
            [
                ("route", "hybrid", "question", "qwen2.5:3b", 0.1),
                ("generate", "question", ["doc"], "qwen2.5:3b", 0.1, "hybrid"),
            ],
        )

    def test_route_question_returns_standard_no_docs_response(self):
        from paper_rag.pipeline.service import route_question

        result = route_question(
            hybrid="hybrid",
            question="question",
            llm_model="qwen2.5:3b",
            temperature=0.1,
            route_retrieve_fn=lambda *_args: ([], "hyde"),
            generate_answer_fn=lambda **_kwargs: "unused",
        )

        self.assertEqual(result, ("❌ 未找到相关内容", []))

    def test_ask_with_hyde_uses_hyde_no_docs_message(self):
        from paper_rag.pipeline.service import ask_with_hyde

        result = ask_with_hyde(
            hybrid="hybrid",
            question="question",
            llm_model="qwen2.5:3b",
            temperature=0.1,
            hyde_retrieve_fn=lambda *_args: [],
            generate_answer_fn=lambda **_kwargs: "unused",
        )

        self.assertEqual(result, ("❌ HyDE 检索未找到相关内容", []))

    def test_ask_with_hyde_generates_answer_from_hyde_docs(self):
        from paper_rag.pipeline.service import ask_with_hyde

        calls = []

        def hyde_retrieve(hybrid, question, llm_model, temperature):
            calls.append(("hyde", hybrid, question, llm_model, temperature))
            return ["hyde-doc"]

        def generate_answer(question, docs, *, llm_model, temperature, hybrid):
            calls.append(("generate", question, docs, llm_model, temperature, hybrid))
            return "hyde answer"

        result = ask_with_hyde(
            hybrid="hybrid",
            question="question",
            llm_model="qwen2.5:3b",
            temperature=0.1,
            hyde_retrieve_fn=hyde_retrieve,
            generate_answer_fn=generate_answer,
        )

        self.assertEqual(result, ("hyde answer", ["hyde-doc"]))
        self.assertEqual(
            calls,
            [
                ("hyde", "hybrid", "question", "qwen2.5:3b", 0.1),
                ("generate", "question", ["hyde-doc"], "qwen2.5:3b", 0.1, "hybrid"),
            ],
        )

    def test_ask_with_context_generates_answer_and_writes_success_log(self):
        from paper_rag.pipeline.service import ask_with_context

        docs = ["doc"]
        calls = []
        logs = []
        notices = []

        def route_retrieve(hybrid, question, llm_model, temperature):
            calls.append(("route", hybrid, question, llm_model, temperature))
            return docs, "mixed"

        def prepare_docs(question, input_docs, *, hybrid, settings):
            calls.append(("prepare", question, input_docs, hybrid, settings))
            return ["prepared-doc"]

        def build_stats(input_docs, context_docs):
            calls.append(("stats", input_docs, context_docs))
            return {"input_chars": 10, "output_chars": 8}

        def generate_answer(question, input_docs, history_text, *, llm_model, temperature, hybrid, prepared_context_docs):
            calls.append(
                (
                    "generate",
                    question,
                    input_docs,
                    history_text,
                    llm_model,
                    temperature,
                    hybrid,
                    prepared_context_docs,
                )
            )
            return "answer"

        result = ask_with_context(
            hybrid="hybrid",
            conversation=FakeConversation(rewritten="standalone question"),
            question="follow up",
            llm_model="qwen2.5:3b",
            temperature=0.1,
            settings="settings",
            route_retrieve_fn=route_retrieve,
            generate_answer_fn=generate_answer,
            write_query_log_fn=lambda **kwargs: logs.append(kwargs),
            prepare_docs_fn=prepare_docs,
            build_stats_fn=build_stats,
            timer_factory=FakeTimer,
            print_fn=notices.append,
        )

        self.assertEqual(result, ("answer", docs))
        self.assertEqual(notices, ['🔄 改写追问: "standalone question"'])
        self.assertEqual(
            calls,
            [
                ("route", "hybrid", "standalone question", "qwen2.5:3b", 0.1),
                ("prepare", "follow up", docs, "hybrid", "settings"),
                ("stats", docs, ["prepared-doc"]),
                (
                    "generate",
                    "follow up",
                    docs,
                    "formatted history",
                    "qwen2.5:3b",
                    0.1,
                    "hybrid",
                    ["prepared-doc"],
                ),
            ],
        )
        self.assertEqual(logs[0]["question"], "follow up")
        self.assertEqual(logs[0]["standalone_question"], "standalone question")
        self.assertEqual(logs[0]["route"], "mixed")
        self.assertEqual(logs[0]["context_stats"], {"input_chars": 10, "output_chars": 8})
        self.assertEqual(logs[0]["elapsed"], {"rewrite": 0.1, "retrieve": 0.2, "generate": 0.3, "total": 0.6})

    def test_ask_with_context_returns_no_docs_response_without_generating(self):
        from paper_rag.pipeline.service import ask_with_context

        logs = []

        result = ask_with_context(
            hybrid="hybrid",
            conversation=FakeConversation(rewritten="standalone question"),
            question="follow up",
            llm_model="qwen2.5:3b",
            temperature=0.1,
            settings="settings",
            route_retrieve_fn=lambda *_args: ([], "hyde"),
            generate_answer_fn=lambda *_args, **_kwargs: self.fail("generation should be skipped"),
            write_query_log_fn=lambda **kwargs: logs.append(kwargs),
            prepare_docs_fn=lambda *_args, **_kwargs: self.fail("context preparation should be skipped"),
            build_stats_fn=lambda *_args: self.fail("context stats should be skipped"),
            timer_factory=FakeTimer,
            print_fn=lambda _message: None,
        )

        self.assertEqual(result, ("❌ 未找到相关内容", []))
        self.assertEqual(logs[0]["docs"], [])
        self.assertEqual(logs[0]["route"], "hyde")
        self.assertEqual(logs[0]["elapsed"], {"rewrite": 0.1, "retrieve": 0.2, "generate": 0.0, "total": 0.3})

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

    def test_handle_llm_unavailable_response_writes_error_log_and_returns_token_event(self):
        from paper_rag.pipeline.service import handle_llm_unavailable_response

        log_calls = []

        events = handle_llm_unavailable_response(
            question="original",
            standalone_question="standalone",
            route="hyde",
            llm_model="qwen2.5:3b",
            docs=["doc"],
            elapsed={"total": 0.4},
            context_stats={"input_chars": 20},
            write_query_log_fn=lambda **kwargs: log_calls.append(kwargs),
        )

        self.assertEqual(events, [{"type": "token", "data": "❌ LLM 模型未连接"}])
        self.assertEqual(log_calls[0]["docs"], ["doc"])
        self.assertEqual(log_calls[0]["context_stats"], {"input_chars": 20})
        self.assertEqual(log_calls[0]["error"], "LLM 模型未连接")

    def test_write_successful_response_log_writes_context_stats_without_error(self):
        from paper_rag.pipeline.service import write_successful_response_log

        log_calls = []

        write_successful_response_log(
            question="original",
            standalone_question="standalone",
            route="mixed",
            llm_model="qwen2.5:3b",
            docs=["doc"],
            elapsed={"total": 0.8},
            context_stats={"input_chars": 20},
            write_query_log_fn=lambda **kwargs: log_calls.append(kwargs),
        )

        self.assertEqual(log_calls[0]["question"], "original")
        self.assertEqual(log_calls[0]["docs"], ["doc"])
        self.assertEqual(log_calls[0]["context_stats"], {"input_chars": 20})
        self.assertNotIn("error", log_calls[0])

    def test_prepare_pipeline_context_returns_prepared_docs_and_stats(self):
        from paper_rag.pipeline.service import prepare_pipeline_context

        calls = []
        original_docs = ["doc"]
        prepared_docs = ["prepared"]

        def prepare_docs(question, docs, *, hybrid, settings):
            calls.append(("prepare", question, docs, hybrid, settings))
            return prepared_docs

        def build_stats(docs, context_docs):
            calls.append(("stats", docs, context_docs))
            return {"input_chars": 10, "output_chars": 4}

        result = prepare_pipeline_context(
            question="original question",
            docs=original_docs,
            hybrid="hybrid",
            settings="settings",
            prepare_docs_fn=prepare_docs,
            build_stats_fn=build_stats,
        )

        self.assertEqual(result.context_docs, prepared_docs)
        self.assertEqual(result.context_stats, {"input_chars": 10, "output_chars": 4})
        self.assertEqual(
            calls,
            [
                ("prepare", "original question", original_docs, "hybrid", "settings"),
                ("stats", original_docs, prepared_docs),
            ],
        )

    def test_stream_token_events_wraps_text_tokens(self):
        from paper_rag.pipeline.service import stream_token_events

        events = list(stream_token_events(["ans", "wer"]))

        self.assertEqual(
            events,
            [
                {"type": "token", "data": "ans"},
                {"type": "token", "data": "wer"},
            ],
        )

    def test_stream_prepared_answer_events_wraps_generation_tokens(self):
        from paper_rag.pipeline.service import stream_prepared_answer_events

        calls = []

        def stream_answer_from_docs(**kwargs):
            calls.append(kwargs)
            return ["ans", "wer"]

        events = list(
            stream_prepared_answer_events(
                question="question",
                docs=["doc"],
                history_text="history",
                llm_model="qwen2.5:3b",
                temperature=0.1,
                hybrid="hybrid",
                settings="settings",
                prepared_context_docs=["prepared"],
                prepare_docs_fn="prepare-docs",
                format_docs_fn="format-docs",
                load_prompt_fn="load-prompt",
                get_llm_fn="get-llm",
                stream_answer_tokens_fn="stream-tokens",
                stream_answer_from_docs_fn=stream_answer_from_docs,
            )
        )

        self.assertEqual(events, [{"type": "token", "data": "ans"}, {"type": "token", "data": "wer"}])
        self.assertEqual(calls[0]["prepared_context_docs"], ["prepared"])
        self.assertEqual(calls[0]["stream_answer_tokens_fn"], "stream-tokens")

    def test_generate_prepared_answer_delegates_to_generation_service(self):
        from paper_rag.pipeline.service import generate_prepared_answer

        calls = []

        def generate_answer_from_docs(**kwargs):
            calls.append(kwargs)
            return "answer"

        answer = generate_prepared_answer(
            question="question",
            docs=["doc"],
            history_text="history",
            llm_model="qwen2.5:3b",
            temperature=0.1,
            hybrid="hybrid",
            settings="settings",
            prepared_context_docs=["prepared"],
            prepare_docs_fn="prepare-docs",
            format_docs_fn="format-docs",
            load_prompt_fn="load-prompt",
            get_llm_fn="get-llm",
            generate_answer_fn="generate-answer",
            generate_answer_from_docs_fn=generate_answer_from_docs,
        )

        self.assertEqual(answer, "answer")
        self.assertEqual(calls[0]["prepared_context_docs"], ["prepared"])
        self.assertEqual(calls[0]["generate_answer_fn"], "generate-answer")

    def test_fixed_llm_factory_ignores_requested_model_and_returns_bound_llm(self):
        from paper_rag.pipeline.service import fixed_llm_factory

        llm = object()
        factory = fixed_llm_factory(llm)

        self.assertIs(factory("other-model", 0.9), llm)

    def test_resolve_stream_llm_delegates_to_runtime_factory(self):
        from paper_rag.pipeline.service import resolve_stream_llm

        calls = []
        llm = object()

        def get_llm(model, temperature):
            calls.append((model, temperature))
            return llm

        self.assertIs(
            resolve_stream_llm(
                llm_model="qwen2.5:3b",
                temperature=0.2,
                get_llm_fn=get_llm,
            ),
            llm,
        )
        self.assertEqual(calls, [("qwen2.5:3b", 0.2)])

    def test_stream_retrieval_events_wraps_route_and_docs(self):
        from paper_rag.pipeline.service import stream_retrieval_events

        docs = ["doc"]
        events = list(stream_retrieval_events("mixed", docs))

        self.assertEqual(
            events,
            [
                {"type": "route", "data": "mixed"},
                {"type": "docs", "data": docs},
            ],
        )

    def test_stream_rewrite_events_wraps_only_rewritten_question(self):
        from paper_rag.pipeline.service import ReformulationResult, stream_rewrite_events

        events = list(stream_rewrite_events(ReformulationResult("standalone", rewritten=True)))
        unchanged_events = list(stream_rewrite_events(ReformulationResult("original", rewritten=False)))

        self.assertEqual(events, [{"type": "rewrite", "data": "standalone"}])
        self.assertEqual(unchanged_events, [])


if __name__ == "__main__":
    unittest.main()
