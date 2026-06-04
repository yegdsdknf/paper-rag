import unittest

from langchain_core.documents import Document

from generation_service import build_rag_prompt, format_docs, generate_answer, stream_answer_tokens


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
