import unittest

from langchain_core.documents import Document


class SourceViewModelTest(unittest.TestCase):
    def test_highlights_english_terms_and_escapes_markup(self):
        from paper_rag.ui.sources import build_source_view_models

        docs = [
            Document(
                page_content="BERT uses masked language modeling. Raw <script> tags stay text.",
                metadata={"source": ".\\papers\\bert.pdf", "page": 2, "rerank_score": 0.8765},
            )
        ]

        views = build_source_view_models(docs, "How does BERT use masked language modeling?")

        self.assertEqual(views[0].title, "bert.pdf · p2")
        self.assertEqual(views[0].source, "bert.pdf")
        self.assertEqual(views[0].page, 2)
        self.assertEqual(views[0].score_label, "rerank 0.877")
        self.assertIn("<mark>BERT</mark>", views[0].highlight_html)
        self.assertIn("<mark>masked</mark>", views[0].highlight_html)
        self.assertIn("&lt;script&gt;", views[0].highlight_html)
        self.assertNotIn("<script>", views[0].highlight_html)

    def test_highlights_chinese_terms_from_question(self):
        from paper_rag.ui.sources import build_source_view_models

        docs = [
            Document(
                page_content="Transformer 架构依赖自注意力机制。位置编码帮助模型理解顺序。",
                metadata={"source": "attention.pdf", "page": 5},
            )
        ]

        views = build_source_view_models(docs, "Transformer 的自注意力机制是什么？")

        self.assertIn("<mark>自注意力机制</mark>", views[0].highlight_html)
        self.assertEqual(views[0].score_label, "")

    def test_uses_best_matching_sentence_with_raw_preview_fallback(self):
        from paper_rag.ui.sources import build_source_view_models

        docs = [
            Document(
                page_content="Intro noise. GPT predicts the next token with causal attention. Appendix noise.",
                metadata={"source": "gpt.pdf", "page": 7, "context_compressed": True},
            )
        ]

        views = build_source_view_models(docs, "next token")

        self.assertEqual(views[0].raw_preview, docs[0].page_content)
        self.assertEqual(views[0].metadata["context_compressed"], True)
        self.assertNotIn("Intro noise", views[0].highlight_html)
        self.assertIn("<mark>next</mark> <mark>token</mark>", views[0].highlight_html)

    def test_handles_empty_metadata_and_no_keyword_match(self):
        from paper_rag.ui.sources import build_source_view_models

        docs = [Document(page_content="plain evidence", metadata={})]

        views = build_source_view_models(docs, "unrelated")

        self.assertEqual(views[0].title, "unknown · p?")
        self.assertEqual(views[0].source, "unknown")
        self.assertEqual(views[0].page, "?")
        self.assertEqual(views[0].highlight_html, "plain evidence")


if __name__ == "__main__":
    unittest.main()
