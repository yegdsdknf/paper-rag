import unittest

from langchain_core.documents import Document


class ChunkMetadataTest(unittest.TestCase):
    def test_attach_chunk_metadata_adds_stable_paper_fields(self):
        from paper_rag.indexing.metadata import attach_chunk_metadata

        docs = [
            Document(
                page_content="Abstract text",
                metadata={"source": "papers/example.pdf", "page": 0, "section_title": "Abstract"},
            ),
            Document(
                page_content="Introduction text",
                metadata={"source": "papers/example.pdf", "page": 1, "section_title": "1 Introduction"},
            ),
        ]

        enriched = attach_chunk_metadata(
            docs,
            strategy="section_aware",
            schema_version="v2",
            source_file_hashes={"example.pdf": "abc123"},
        )

        self.assertEqual(enriched[0].metadata["doc_id"], "doc_abc123")
        self.assertEqual(enriched[0].metadata["source_file"], "example.pdf")
        self.assertEqual(enriched[0].metadata["global_chunk_index"], 0)
        self.assertEqual(enriched[0].metadata["page_chunk_index"], 0)
        self.assertEqual(enriched[1].metadata["page_chunk_index"], 0)
        self.assertEqual(enriched[0].metadata["chunk_strategy"], "section_aware")
        self.assertEqual(enriched[0].metadata["chunk_schema_version"], "v2")
        self.assertEqual(enriched[0].metadata["paper_region"], "abstract")
        self.assertTrue(enriched[0].metadata["chunk_id"].startswith("doc_abc123:v2:"))
        self.assertEqual(docs[0].metadata.get("doc_id"), None)

    def test_attach_chunk_metadata_orders_multiple_chunks_on_same_page(self):
        from paper_rag.indexing.metadata import attach_chunk_metadata

        docs = [
            Document(page_content="first", metadata={"source": "paper.pdf", "page": 3}),
            Document(page_content="second", metadata={"source": "paper.pdf", "page": 3}),
            Document(page_content="third", metadata={"source": "paper.pdf", "page": 4}),
        ]

        enriched = attach_chunk_metadata(docs, strategy="recursive_character", schema_version="v2")

        self.assertEqual([doc.metadata["page_chunk_index"] for doc in enriched], [0, 1, 0])


if __name__ == "__main__":
    unittest.main()
