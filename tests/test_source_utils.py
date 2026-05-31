import unittest

from langchain_core.documents import Document

from source_utils import source_from_doc, sources_from_docs


class SourceUtilsTest(unittest.TestCase):
    def test_source_from_doc_normalizes_file_page_preview_and_rerank_score(self):
        doc = Document(
            page_content="x" * 260,
            metadata={"source": ".\\papers\\bert.pdf", "page": 2, "rerank_score": 0.91},
        )

        source = source_from_doc(doc, preview_chars=200)

        self.assertEqual(source["file"], "bert.pdf")
        self.assertEqual(source["page"], 2)
        self.assertEqual(len(source["content_preview"]), 200)
        self.assertEqual(source["rerank_score"], 0.91)

    def test_source_from_doc_uses_stable_defaults_for_missing_metadata(self):
        doc = Document(page_content="evidence", metadata={})

        source = source_from_doc(doc)

        self.assertEqual(source["file"], "unknown")
        self.assertEqual(source["page"], -1)
        self.assertEqual(source["content_preview"], "evidence")
        self.assertNotIn("rerank_score", source)

    def test_sources_from_docs_applies_same_schema_to_all_documents(self):
        docs = [
            Document(page_content="first", metadata={"source": "a.pdf", "page": 0}),
            Document(page_content="second", metadata={"source": "b.pdf", "page": 1}),
        ]

        sources = sources_from_docs(docs, preview_chars=3)

        self.assertEqual(
            sources,
            [
                {"file": "a.pdf", "page": 0, "content_preview": "fir"},
                {"file": "b.pdf", "page": 1, "content_preview": "sec"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
