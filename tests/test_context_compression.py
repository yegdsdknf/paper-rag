import unittest
from unittest.mock import patch

from langchain_core.documents import Document

import rag_pipeline
from context_compression import compress_chunk, compress_documents


class CapturingLLM:
    def __init__(self):
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return type("Response", (), {"content": "answer"})()


class ContextCompressionTest(unittest.TestCase):
    def test_compress_chunk_keeps_query_relevant_sentences_and_drops_noise(self):
        text = (
            "This paper introduces unrelated training details. "
            "The Transformer is based solely on attention mechanisms. "
            "Appendix tables describe optimizer settings."
        )

        compressed = compress_chunk("Transformer attention mechanism", text, max_sentences=2)

        self.assertIn("based solely on attention mechanisms", compressed)
        self.assertNotIn("optimizer settings", compressed)
        self.assertLess(len(compressed), len(text))

    def test_compress_chunk_falls_back_to_original_when_no_sentence_matches(self):
        text = "Alpha beta gamma. Delta epsilon."

        self.assertEqual(compress_chunk("vision transformer patches", text), text)

    def test_compress_documents_preserves_metadata_and_records_lengths(self):
        docs = [
            Document(
                page_content="Noise sentence. BERT uses masked language modeling. More noise.",
                metadata={"source": "bert.pdf", "page": 1},
            )
        ]

        compressed_docs = compress_documents("masked language modeling", docs, max_sentences=1)

        self.assertEqual(compressed_docs[0].metadata["source"], "bert.pdf")
        self.assertTrue(compressed_docs[0].metadata["context_compressed"])
        self.assertGreater(
            compressed_docs[0].metadata["context_original_chars"],
            compressed_docs[0].metadata["context_compressed_chars"],
        )
        self.assertEqual(compressed_docs[0].page_content, "BERT uses masked language modeling.")

    def test_generate_answer_compresses_prompt_context_without_mutating_sources(self):
        docs = [
            Document(
                page_content=(
                    "Long unrelated preface. "
                    "The Transformer is based solely on attention mechanisms. "
                    "Long unrelated appendix."
                ),
                metadata={"source": "attention.pdf", "page": 0},
            )
        ]
        llm = CapturingLLM()

        with (
            patch.dict(
                rag_pipeline.config,
                {
                    "enable_context_compression": True,
                    "context_compression_max_sentences": 1,
                },
            ),
            patch.object(rag_pipeline, "_get_llm", return_value=llm),
        ):
            answer = rag_pipeline._generate_answer("What is Transformer based on?", docs)

        self.assertEqual(answer, "answer")
        self.assertEqual(
            docs[0].page_content,
            "Long unrelated preface. The Transformer is based solely on attention mechanisms. Long unrelated appendix.",
        )
        self.assertIn("based solely on attention mechanisms", llm.prompts[0])
        self.assertNotIn("Long unrelated appendix", llm.prompts[0])


if __name__ == "__main__":
    unittest.main()
