import unittest

from langchain_core.documents import Document

from paper_rag.config import RagSettings


class FakeEmbeddings:
    def embed_documents(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "apple" in lowered or "banana" in lowered:
                vectors.append([1.0, 0.0])
            elif "transformer" in lowered or "attention" in lowered:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.5, 0.5])
        return vectors


def make_settings(**overrides):
    data = {
        "persist_directory": "./chroma_db",
        "embedding_model": "BAAI/bge-m3",
        "llm_model": "qwen2.5:3b",
        "temperature": 0.1,
        "k": 6,
        "chunk_size": 500,
        "chunk_overlap": 50,
        "separators": ["\n\n", "\n", ". ", " "],
        "chunk_strategy": "semantic",
        "chunk_schema_version": "v3",
        "semantic_similarity_threshold": 0.8,
        "semantic_min_chars": 10,
        "semantic_max_chars": 180,
    }
    data.update(overrides)
    return RagSettings.from_mapping(data)


class SemanticChunkingTest(unittest.TestCase):
    def test_semantic_split_breaks_when_adjacent_sentence_similarity_is_low(self):
        from paper_rag.indexing.chunking import split_documents

        doc = Document(
            page_content=(
                "Apple banana fruit results are discussed. "
                "Apple banana experiments remain related. "
                "Transformer attention layers are introduced."
            ),
            metadata={"source": "papers/example.pdf", "page": 2},
        )

        chunks = split_documents([doc], make_settings(), embeddings=FakeEmbeddings())

        self.assertEqual(len(chunks), 2)
        self.assertIn("Apple banana experiments", chunks[0].page_content)
        self.assertIn("Transformer attention", chunks[1].page_content)
        self.assertEqual(chunks[0].metadata["chunk_strategy"], "semantic")

    def test_semantic_split_requires_embeddings(self):
        from paper_rag.indexing.chunking import split_documents

        doc = Document(page_content="A sentence. Another sentence.", metadata={"source": "paper.pdf", "page": 0})

        with self.assertRaises(ValueError):
            split_documents([doc], make_settings(), embeddings=None)

    def test_semantic_split_recursively_splits_single_oversized_sentence(self):
        from paper_rag.indexing.chunking import split_documents

        long_table_row = " ".join(f"value{i}" for i in range(160))
        doc = Document(page_content=long_table_row, metadata={"source": "paper.pdf", "page": 62})

        chunks = split_documents(
            [doc],
            make_settings(semantic_max_chars=180, chunk_size=180, chunk_overlap=20),
            embeddings=FakeEmbeddings(),
        )

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk.page_content) <= 220 for chunk in chunks))
        self.assertTrue(all(chunk.metadata["chunk_strategy"] == "semantic" for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
