import unittest


class FriendlyErrorTest(unittest.TestCase):
    def test_formats_ollama_connection_error_with_doctor_hint(self):
        from paper_rag.ui.errors import format_runtime_error

        error = format_runtime_error(RuntimeError("Connection refused"))

        self.assertEqual(error.title, "无法连接到 Ollama 服务")
        self.assertTrue(error.show_doctor_hint)
        self.assertIn("ollama serve", "\n".join(error.suggestions))

    def test_formats_ollama_missing_model_with_pull_command(self):
        from paper_rag.ui.errors import format_runtime_error

        error = format_runtime_error(RuntimeError("model not found"), {"llm_model": "qwen2.5:3b"})

        self.assertEqual(error.title, "Ollama 模型未下载")
        self.assertFalse(error.show_doctor_hint)
        self.assertIn("ollama pull qwen2.5:3b", "\n".join(error.suggestions))

    def test_formats_local_embedding_error(self):
        from paper_rag.ui.errors import format_runtime_error

        error = format_runtime_error(RuntimeError("local model not found: BAAI/bge-m3"))

        self.assertEqual(error.title, "本地 Embedding 模型不可用")
        self.assertTrue(error.show_doctor_hint)

    def test_formats_chroma_empty_or_missing_error(self):
        from paper_rag.ui.errors import format_runtime_error

        error = format_runtime_error(RuntimeError("collection is empty"))

        self.assertEqual(error.title, "向量库未构建或为空")
        self.assertIn("python main.py build", "\n".join(error.suggestions))

    def test_formats_reranker_unavailable_as_warning(self):
        from paper_rag.ui.errors import format_runtime_error

        error = format_runtime_error(RuntimeError("Rerank unavailable; keeping original order"))

        self.assertEqual(error.title, "精排模型不可用")
        self.assertEqual(error.severity, "warning")
        self.assertIn("原始检索顺序", error.message)

    def test_fallback_keeps_short_details_and_doctor_hint(self):
        from paper_rag.ui.errors import format_runtime_error

        error = format_runtime_error(ValueError("unexpected internal failure"))

        self.assertEqual(error.title, "运行时错误")
        self.assertTrue(error.show_doctor_hint)
        self.assertIn("unexpected internal failure", error.details)

    def test_render_streamlit_error_uses_structured_sections(self):
        from paper_rag.ui.errors import FriendlyError, render_streamlit_error

        class FakeStreamlit:
            def __init__(self):
                self.calls = []

            def error(self, text):
                self.calls.append(("error", text))

            def warning(self, text):
                self.calls.append(("warning", text))

            def markdown(self, text):
                self.calls.append(("markdown", text))

            def expander(self, title):
                self.calls.append(("expander", title))
                return self

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        fake = FakeStreamlit()
        render_streamlit_error(
            fake,
            FriendlyError(
                title="无法连接到 Ollama 服务",
                message="服务不可用",
                suggestions=["运行 ollama serve"],
                details="Connection refused",
            ),
        )

        self.assertIn(("error", "无法连接到 Ollama 服务"), fake.calls)
        self.assertIn(("markdown", "- 运行 ollama serve"), fake.calls)
        self.assertIn(("expander", "技术详情"), fake.calls)


if __name__ == "__main__":
    unittest.main()
