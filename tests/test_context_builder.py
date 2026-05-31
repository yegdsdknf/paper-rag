import unittest

from langchain_core.documents import Document

from context_builder import build_context_stats, prepare_docs_for_context


class FakeVectorStore:
    def get(self, include=None):
        return {
            "documents": [
                "page 1 opening context",
                "page 1 retrieved child with attention",
                "page 1 closing context",
                "page 2 unrelated",
            ],
            "metadatas": [
                {"source": "paper.pdf", "page": 1},
                {"source": "paper.pdf", "page": 1},
                {"source": "paper.pdf", "page": 1},
                {"source": "paper.pdf", "page": 2},
            ],
        }


class FakeHybrid:
    vector_store = FakeVectorStore()


class ContextBuilderTest(unittest.TestCase):
    def test_prepare_docs_returns_original_docs_when_features_disabled(self):
        docs = [Document(page_content="child", metadata={"source": "paper.pdf", "page": 1})]

        context_docs = prepare_docs_for_context(
            "question",
            docs,
            hybrid=FakeHybrid(),
            settings={
                "enable_parent_retrieval": False,
                "enable_context_compression": False,
            },
        )

        self.assertIs(context_docs, docs)

    def test_prepare_docs_expands_parent_before_compression(self):
        docs = [
            Document(
                page_content="page 1 retrieved child with attention",
                metadata={"source": "paper.pdf", "page": 1},
            )
        ]

        context_docs = prepare_docs_for_context(
            "attention",
            docs,
            hybrid=FakeHybrid(),
            settings={
                "enable_parent_retrieval": True,
                "parent_max_chars_per_page": 200,
                "enable_context_compression": True,
                "context_compression_max_sentences": 1,
            },
        )

        self.assertEqual(len(context_docs), 1)
        self.assertTrue(context_docs[0].metadata["parent_context"])
        self.assertTrue(context_docs[0].metadata["context_compressed"])
        self.assertIn("retrieved child with attention", context_docs[0].page_content)
        self.assertNotIn("page 2 unrelated", context_docs[0].page_content)
        self.assertEqual(docs[0].page_content, "page 1 retrieved child with attention")

    def test_prepare_docs_accepts_typed_settings(self):
        from paper_rag.config import RagSettings

        docs = [
            Document(
                page_content="page 1 retrieved child with attention",
                metadata={"source": "paper.pdf", "page": 1},
            )
        ]
        settings = RagSettings.from_mapping(
            {
                "persist_directory": "./chroma_db",
                "embedding_model": "BAAI/bge-m3",
                "llm_model": "qwen2.5:3b",
                "temperature": 0.1,
                "k": 6,
                "chunk_size": 500,
                "chunk_overlap": 100,
                "separators": ["\n\n", "\n"],
                "enable_parent_retrieval": True,
                "parent_max_chars_per_page": 80,
                "enable_context_compression": False,
            }
        )

        context_docs = prepare_docs_for_context("attention", docs, hybrid=FakeHybrid(), settings=settings)

        self.assertTrue(context_docs[0].metadata["parent_context"])
        self.assertGreater(len(context_docs[0].page_content), len(docs[0].page_content))

    def test_build_context_stats_counts_docs_and_chars(self):
        original_docs = [
            Document(page_content="aaaa", metadata={}),
            Document(page_content="bbbbbb", metadata={}),
        ]
        context_docs = [Document(page_content="ccc", metadata={})]

        stats = build_context_stats(original_docs, context_docs)

        self.assertEqual(
            stats,
            {
                "source_doc_count": 2,
                "context_doc_count": 1,
                "input_chars": 10,
                "output_chars": 3,
            },
        )


if __name__ == "__main__":
    unittest.main()
