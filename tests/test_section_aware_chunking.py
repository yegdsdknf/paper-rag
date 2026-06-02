import unittest

from langchain_core.documents import Document

from paper_rag.config import RagSettings


def make_settings(**overrides):
    data = {
        "persist_directory": "./chroma_db",
        "embedding_model": "BAAI/bge-m3",
        "llm_model": "qwen2.5:3b",
        "temperature": 0.1,
        "k": 6,
        "chunk_size": 120,
        "chunk_overlap": 20,
        "separators": ["\n\n", "\n", ". ", " "],
        "chunk_strategy": "section_aware",
        "chunk_schema_version": "v2",
        "section_max_chars": 260,
    }
    data.update(overrides)
    return RagSettings.from_mapping(data)


class SectionAwareChunkingTest(unittest.TestCase):
    def test_section_aware_keeps_abstract_and_detects_numbered_headings(self):
        from paper_rag.indexing.chunking import split_documents

        page = Document(
            page_content=(
                "A Paper Title\n"
                "Alice Example\n\n"
                "Abstract\n"
                "This paper studies retrieval augmented generation for scientific papers.\n"
                "The abstract should remain together.\n\n"
                "1 Introduction\n"
                "Retrieval systems need chunks that respect section boundaries.\n\n"
                "2 Method\n"
                "We combine paper structure with fallback recursive splitting."
            ),
            metadata={"source": "papers/example.pdf", "page": 0},
        )

        chunks = split_documents([page], make_settings())
        titles = [chunk.metadata.get("section_title") for chunk in chunks]

        self.assertIn("Abstract", titles)
        self.assertIn("1 Introduction", titles)
        self.assertIn("2 Method", titles)
        abstract_chunks = [chunk for chunk in chunks if chunk.metadata.get("section_title") == "Abstract"]
        self.assertEqual(len(abstract_chunks), 1)
        self.assertIn("abstract should remain together", abstract_chunks[0].page_content)
        self.assertEqual(abstract_chunks[0].metadata["paper_region"], "abstract")

    def test_section_aware_falls_back_to_recursive_for_long_sections(self):
        from paper_rag.indexing.chunking import split_documents

        long_body = " ".join(f"sentence{i}" for i in range(80))
        page = Document(
            page_content=f"1 Introduction\n{long_body}",
            metadata={"source": "papers/example.pdf", "page": 1},
        )

        chunks = split_documents([page], make_settings(section_max_chars=160))

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.metadata.get("section_title") == "1 Introduction" for chunk in chunks))
        self.assertTrue(all(len(chunk.page_content) <= 180 for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
