import unittest

from langchain_core.documents import Document


class PdfTextFilterTest(unittest.TestCase):
    def test_detects_pages_dominated_by_embedded_font_unicode_codes(self):
        from paper_rag.indexing.pdf_text import is_noisy_pdf_text

        noisy_text = " ".join(f"/uni000000{i % 10:02d}" for i in range(120))

        self.assertTrue(is_noisy_pdf_text(noisy_text))

    def test_keeps_normal_paper_text_with_a_few_unicode_codes(self):
        from paper_rag.indexing.pdf_text import is_noisy_pdf_text

        text = "This page discusses safety evaluation and includes one token /uni00000019 only."

        self.assertFalse(is_noisy_pdf_text(text))

    def test_filter_noisy_pages_preserves_clean_documents_and_reports_skipped_pages(self):
        from paper_rag.indexing.pdf_text import filter_noisy_pdf_pages

        docs = [
            Document(page_content="Clean abstract text", metadata={"source": "paper.pdf", "page": 0}),
            Document(
                page_content=" ".join("/uni00000019" for _ in range(100)),
                metadata={"source": "paper.pdf", "page": 1},
            ),
        ]

        kept, skipped = filter_noisy_pdf_pages(docs)

        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].metadata["page"], 0)
        self.assertEqual(skipped, [{"source": "paper.pdf", "page": 1, "reason": "unicode_escape_noise"}])

    def test_analyze_pdf_text_quality_adds_flags_without_mutating_input(self):
        from paper_rag.indexing.pdf_text import analyze_pdf_text_quality

        docs = [
            Document(page_content="Clean abstract text", metadata={"source": "paper.pdf", "page": 0}),
            Document(
                page_content=" ".join("/uni00000019" for _ in range(100)),
                metadata={"source": "paper.pdf", "page": 1},
            ),
        ]

        clean_docs, reports = analyze_pdf_text_quality(docs)

        self.assertEqual(clean_docs[0].metadata["quality_flags"], ["text_quality_checked"])
        self.assertEqual(reports[0]["quality_flags"], ["unicode_escape_noise"])
        self.assertEqual(docs[0].metadata.get("quality_flags"), None)


if __name__ == "__main__":
    unittest.main()
