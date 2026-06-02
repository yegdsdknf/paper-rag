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

    def test_compress_documents_preserves_vision_summary_details(self):
        vision_text = (
            "1. 页面类型：figure\n"
            "2. 图表编号或标题：Figure 14 | Multilingual safety performance.\n"
            "3. 主要内容：图14展示了 DeepSeek-V3 和 DeepSeek-R1 的 multilingual safety performance。\n"
            "4. 关键指标/数值/趋势：丹麦语在 V3-check 下约为77.6，在 R1-check 下约为87.6；"
            "乌克兰语和乌兹别克语也展示了不同安全评分趋势。\n"
            "5. 可用于回答的问题：DeepSeek-R1 在多语言安全评估中表现如何？"
        )
        docs = [
            Document(
                page_content=vision_text,
                metadata={"source": "deepseekr1.pdf", "page": 51, "block_type": "vision_summary"},
            )
        ]

        compressed_docs = compress_documents("DeepSeek-R1 multilingual safety performance", docs, max_sentences=1)

        self.assertEqual(compressed_docs[0].page_content, vision_text)
        self.assertFalse(compressed_docs[0].metadata["context_compressed"])
        self.assertIn("丹麦语", compressed_docs[0].page_content)
        self.assertIn("87.6", compressed_docs[0].page_content)

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
