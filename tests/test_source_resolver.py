import tempfile
import unittest
from pathlib import Path

from langchain_core.documents import Document


class FakeVectorStore:
    def __init__(self, rows):
        self.rows = rows

    def get(self, include=None):
        return {
            "documents": [content for content, _metadata in self.rows],
            "metadatas": [metadata for _content, metadata in self.rows],
        }


class FakeHybrid:
    def __init__(self, rows):
        self.vector_store = FakeVectorStore(rows)


class SourceResolverTest(unittest.TestCase):
    def test_resolves_new_filename_alias_without_router_code_change(self):
        from paper_rag.retrieval.source_resolver import SourceResolver

        resolver = SourceResolver.from_hybrid(
            FakeHybrid(
                [
                    (
                        "MiniLM learns compact sentence representations.",
                        {"source": "MiniLM-L6.pdf", "page": 0},
                    )
                ]
            )
        )

        self.assertEqual(resolver.resolve_source_files("MiniLM-L6 的核心思路是什么？"), ["MiniLM-L6.pdf"])
        self.assertEqual(resolver.resolve_source_files("minilm l6 uses what objective?"), ["MiniLM-L6.pdf"])

    def test_generic_transformer_origin_uses_front_page_content_not_static_alias(self):
        from paper_rag.retrieval.source_resolver import SourceResolver

        resolver = SourceResolver.from_hybrid(
            FakeHybrid(
                [
                    (
                        "Attention Is All You Need. We propose a new simple network architecture, the Transformer.",
                        {"source": "attention is all you need.pdf", "page": 0},
                    ),
                    (
                        "BERT: Bidirectional Encoder Representations from Transformers.",
                        {"source": "bert.pdf", "page": 0},
                    ),
                ]
            )
        )

        self.assertEqual(resolver.resolve_source_files("Transformer 是基于什么机制构建的？"), ["attention is all you need.pdf"])

    def test_origin_detection_uses_multiple_front_page_chunks(self):
        from paper_rag.retrieval.source_resolver import SourceResolver

        resolver = SourceResolver.from_hybrid(
            FakeHybrid(
                [
                    (
                        "Attention Is All You Need. Author list and affiliations.",
                        {"source": "attention is all you need.pdf", "page": 0},
                    ),
                    (
                        "Abstract. We propose a new simple network architecture, the Transformer.",
                        {"source": "attention is all you need.pdf", "page": 0},
                    ),
                    (
                        "BERT: Bidirectional Encoder Representations from Transformers.",
                        {"source": "bert.pdf", "page": 0},
                    ),
                ]
            )
        )

        self.assertEqual(resolver.resolve_source_files("Transformer 是什么？"), ["attention is all you need.pdf"])

    def test_explicit_source_prevents_generic_transformer_origin_expansion(self):
        from paper_rag.retrieval.source_resolver import SourceResolver

        resolver = SourceResolver.from_hybrid(
            FakeHybrid(
                [
                    (
                        "Attention Is All You Need. We propose a new simple network architecture, the Transformer.",
                        {"source": "attention is all you need.pdf", "page": 0},
                    ),
                    (
                        "BERT: Bidirectional Encoder Representations from Transformers.",
                        {"source": "bert.pdf", "page": 0},
                    ),
                ]
            )
        )

        self.assertEqual(resolver.resolve_source_files("BERT 为什么叫双向 Transformer？"), ["bert.pdf"])

    def test_manifest_sources_are_used_when_vector_store_is_unavailable(self):
        from paper_rag.config import RagSettings
        from paper_rag.indexing.manifest import save_index_manifest
        from paper_rag.retrieval.source_resolver import SourceResolver

        with tempfile.TemporaryDirectory() as tmp:
            settings = RagSettings.from_mapping(
                {
                    "persist_directory": tmp,
                    "embedding_model": "BAAI/bge-m3",
                    "llm_model": "qwen2.5:3b",
                    "temperature": 0.1,
                    "k": 6,
                    "chunk_size": 500,
                    "chunk_overlap": 100,
                    "separators": ["\n\n", "\n"],
                }
            )
            save_index_manifest(
                {
                    "index_version": "idx_test",
                    "source_files": [{"filename": "LongNet-Scaling-Transformers.pdf", "path": str(Path(tmp) / "LongNet-Scaling-Transformers.pdf")}],
                },
                settings,
            )

            resolver = SourceResolver.from_hybrid(None, settings=settings)

        self.assertEqual(
            resolver.resolve_source_files("LongNet Scaling Transformers 的贡献是什么？"),
            ["LongNet-Scaling-Transformers.pdf"],
        )


if __name__ == "__main__":
    unittest.main()
