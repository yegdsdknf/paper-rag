import unittest
from pathlib import Path


class RagPromptTest(unittest.TestCase):
    def test_prompt_instructs_model_to_prioritize_vision_summary_details(self):
        prompt = Path("prompts/rag_summary_prompt.txt").read_text(encoding="utf-8")

        self.assertIn("vision_summary", prompt)
        self.assertIn("图表编号", prompt)
        self.assertIn("关键数值", prompt)
        self.assertIn("趋势", prompt)


if __name__ == "__main__":
    unittest.main()
