import unittest

from langchain_core.documents import Document

from generation_service import (
    build_rag_prompt,
    format_docs,
    generate_answer,
    generate_answer_from_docs,
    stream_answer_from_docs,
    stream_answer_tokens,
)


class FakeLLM:
    def __init__(self):
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return type("Response", (), {"content": " answer "})()

    def stream(self, prompt):
        self.prompts.append(prompt)
        yield type("Chunk", (), {"content": "ans"})()
        yield type("Chunk", (), {"content": "wer"})()


class StringLLM:
    def invoke(self, prompt):
        return " plain answer "

    def stream(self, prompt):
        yield "plain "
        yield "stream"


class GenerationServiceTest(unittest.TestCase):
    def test_build_rag_prompt_includes_history_instruction_context_and_question(self):
        prompt = build_rag_prompt(
            prompt_template="Context: {context}\nQuestion: {question}",
            context="doc context",
            question="What is BERT?",
            history_text="history\n",
        )

        self.assertTrue(prompt.startswith("history\n请严格按"))
        self.assertIn("Context: doc context", prompt)
        self.assertIn("Question: What is BERT?", prompt)

    def test_format_docs_includes_source_page_and_separator(self):
        docs = [
            Document(page_content="first chunk", metadata={"source": "paper-a.pdf", "page": 3}),
            Document(page_content="second chunk", metadata={}),
        ]

        context = format_docs(docs)

        self.assertEqual(
            context,
            "[片段1 | 来源=paper-a.pdf | 页码=3]\nfirst chunk"
            "\n\n---\n\n"
            "[片段2 | 来源=未知来源 | 页码=?]\nsecond chunk",
        )

    def test_generate_answer_invokes_llm_and_strips_content(self):
        llm = FakeLLM()

        answer = generate_answer(
            llm,
            prompt_template="Context: {context}\nQuestion: {question}",
            context="doc context",
            question="What is BERT?",
        )

        self.assertEqual(answer, "answer")
        self.assertEqual(len(llm.prompts), 1)

    def test_generate_answer_from_docs_prepares_context_and_invokes_llm(self):
        calls = []
        docs = [Document(page_content="raw", metadata={"source": "paper.pdf", "page": 1})]
        prepared_docs = [Document(page_content="prepared", metadata={"source": "paper.pdf", "page": 1})]
        llm = FakeLLM()

        def prepare_docs(question, input_docs, *, hybrid, settings):
            calls.append(("prepare", question, input_docs, hybrid, settings))
            return prepared_docs

        def format_context(input_docs):
            calls.append(("format", input_docs))
            return "formatted context"

        def load_prompt(name):
            calls.append(("prompt", name))
            return "Context: {context}\nQuestion: {question}"

        def get_llm(model, temperature):
            calls.append(("llm", model, temperature))
            return llm

        answer = generate_answer_from_docs(
            question="What is BERT?",
            docs=docs,
            history_text="history\n",
            llm_model="qwen2.5:3b",
            temperature=0.2,
            hybrid="hybrid",
            settings={"enable_context_compression": False},
            prepare_docs_fn=prepare_docs,
            format_docs_fn=format_context,
            load_prompt_fn=load_prompt,
            get_llm_fn=get_llm,
        )

        self.assertEqual(answer, "answer")
        self.assertEqual(calls[0], ("prepare", "What is BERT?", docs, "hybrid", {"enable_context_compression": False}))
        self.assertEqual(calls[1], ("format", prepared_docs))
        self.assertEqual(calls[2], ("prompt", "rag_summary_prompt"))
        self.assertEqual(calls[3], ("llm", "qwen2.5:3b", 0.2))
        self.assertIn("history\n请严格按", llm.prompts[0])
        self.assertIn("formatted context", llm.prompts[0])

    def test_generate_answer_returns_connection_message_when_llm_missing(self):
        answer = generate_answer(
            None,
            prompt_template="Context: {context}\nQuestion: {question}",
            context="doc context",
            question="What is BERT?",
        )

        self.assertEqual(answer, "❌ LLM 模型未连接，请检查 Ollama 服务")

    def test_stream_answer_tokens_yields_text_from_chunks(self):
        tokens = list(
            stream_answer_tokens(
                FakeLLM(),
                prompt_template="Context: {context}\nQuestion: {question}",
                context="doc context",
                question="What is BERT?",
            )
        )

        self.assertEqual(tokens, ["ans", "wer"])

    def test_stream_answer_from_docs_prepares_context_and_streams_tokens(self):
        calls = []
        docs = [Document(page_content="raw", metadata={"source": "paper.pdf", "page": 1})]
        prepared_docs = [Document(page_content="prepared", metadata={"source": "paper.pdf", "page": 1})]
        llm = FakeLLM()

        def prepare_docs(question, input_docs, *, hybrid, settings):
            calls.append(("prepare", question, input_docs, hybrid, settings))
            return prepared_docs

        def format_context(input_docs):
            calls.append(("format", input_docs))
            return "formatted context"

        def load_prompt(name):
            calls.append(("prompt", name))
            return "Context: {context}\nQuestion: {question}"

        def get_llm(model, temperature):
            calls.append(("llm", model, temperature))
            return llm

        tokens = list(
            stream_answer_from_docs(
                question="What is BERT?",
                docs=docs,
                history_text="history\n",
                llm_model="qwen2.5:3b",
                temperature=0.2,
                hybrid="hybrid",
                settings={"enable_context_compression": False},
                prepare_docs_fn=prepare_docs,
                format_docs_fn=format_context,
                load_prompt_fn=load_prompt,
                get_llm_fn=get_llm,
            )
        )

        self.assertEqual(tokens, ["ans", "wer"])
        self.assertEqual(calls[0], ("prepare", "What is BERT?", docs, "hybrid", {"enable_context_compression": False}))
        self.assertEqual(calls[1], ("format", prepared_docs))
        self.assertEqual(calls[2], ("prompt", "rag_summary_prompt"))
        self.assertEqual(calls[3], ("llm", "qwen2.5:3b", 0.2))
        self.assertIn("history\n请严格按", llm.prompts[0])
        self.assertIn("formatted context", llm.prompts[0])

    def test_stream_answer_tokens_supports_plain_string_chunks(self):
        tokens = list(
            stream_answer_tokens(
                StringLLM(),
                prompt_template="Context: {context}\nQuestion: {question}",
                context="doc context",
                question="What is BERT?",
            )
        )

        self.assertEqual(tokens, ["plain ", "stream"])


if __name__ == "__main__":
    unittest.main()
