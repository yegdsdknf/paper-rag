import unittest
from unittest.mock import patch

from langchain_core.documents import Document

import rag_pipeline
from parent_retrieval import expand_parent_pages


class FakeVectorStore:
    def get(self, include=None):
        return {
            "documents": [
                "page 1 opening context",
                "page 1 retrieved child",
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


class CapturingLLM:
    def __init__(self):
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return type("Response", (), {"content": "answer"})()


class ParentRetrievalTest(unittest.TestCase):
    def test_expand_parent_pages_merges_chunks_from_same_source_page(self):
        child_docs = [
            Document(
                page_content="page 1 retrieved child",
                metadata={"source": "paper.pdf", "page": 1, "rerank_score": 0.9},
            )
        ]

        parent_docs = expand_parent_pages(FakeHybrid(), child_docs, max_chars_per_parent=200)

        self.assertEqual(len(parent_docs), 1)
        self.assertIn("page 1 opening context", parent_docs[0].page_content)
        self.assertIn("page 1 retrieved child", parent_docs[0].page_content)
        self.assertIn("page 1 closing context", parent_docs[0].page_content)
        self.assertNotIn("page 2 unrelated", parent_docs[0].page_content)
        self.assertEqual(parent_docs[0].metadata["source"], "paper.pdf")
        self.assertEqual(parent_docs[0].metadata["page"], 1)
        self.assertTrue(parent_docs[0].metadata["parent_context"])

    def test_expand_parent_pages_uses_page_chunk_index_order(self):
        class ShuffledVectorStore:
            def get(self, include=None):
                return {
                    "documents": ["middle", "opening", "closing"],
                    "metadatas": [
                        {"source": "paper.pdf", "page": 1, "page_chunk_index": 1},
                        {"source": "paper.pdf", "page": 1, "page_chunk_index": 0},
                        {"source": "paper.pdf", "page": 1, "page_chunk_index": 2},
                    ],
                }

        class ShuffledHybrid:
            vector_store = ShuffledVectorStore()

        child_docs = [
            Document(page_content="middle", metadata={"source": "paper.pdf", "page": 1})
        ]

        parent_docs = expand_parent_pages(ShuffledHybrid(), child_docs, max_chars_per_parent=200)

        self.assertEqual(parent_docs[0].page_content, "opening\n\nmiddle\n\nclosing")

    def test_expand_parent_pages_preserves_vision_summary_child(self):
        class VisionVectorStore:
            def get(self, include=None):
                return {
                    "documents": ["Figure 14 vision summary with Danish 87.6 score"],
                    "metadatas": [
                        {
                            "source": "deepseekr1.pdf",
                            "page": 51,
                            "block_type": "vision_summary",
                            "chunk_strategy": "vision_summary",
                        }
                    ],
                }

        class VisionHybrid:
            vector_store = VisionVectorStore()

        child_docs = [
            Document(
                page_content="Figure 14 vision summary with Danish 87.6 score",
                metadata={
                    "source": "deepseekr1.pdf",
                    "page": 51,
                    "block_type": "vision_summary",
                    "chunk_strategy": "vision_summary",
                },
            )
        ]

        parent_docs = expand_parent_pages(VisionHybrid(), child_docs, max_chars_per_parent=200)

        self.assertEqual(len(parent_docs), 1)
        self.assertEqual(parent_docs[0].page_content, child_docs[0].page_content)
        self.assertEqual(parent_docs[0].metadata["block_type"], "vision_summary")
        self.assertNotIn("parent_context", parent_docs[0].metadata)

    def test_generate_answer_uses_parent_context_without_mutating_sources(self):
        child_docs = [
            Document(
                page_content="page 1 retrieved child",
                metadata={"source": "paper.pdf", "page": 1},
            )
        ]
        llm = CapturingLLM()

        with (
            patch.dict(
                rag_pipeline.config,
                {
                    "enable_parent_retrieval": True,
                    "parent_max_chars_per_page": 200,
                    "enable_context_compression": False,
                },
            ),
            patch.object(rag_pipeline, "_get_llm", return_value=llm),
        ):
            answer = rag_pipeline._generate_answer(
                "需要完整页面上下文的问题",
                child_docs,
                hybrid=FakeHybrid(),
            )

        self.assertEqual(answer, "answer")
        self.assertEqual(child_docs[0].page_content, "page 1 retrieved child")
        self.assertIn("page 1 opening context", llm.prompts[0])
        self.assertIn("page 1 closing context", llm.prompts[0])


if __name__ == "__main__":
    unittest.main()
