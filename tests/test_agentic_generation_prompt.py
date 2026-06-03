import unittest


class AgenticGenerationPromptTest(unittest.TestCase):
    def test_build_rag_prompt_places_verified_summary_before_context(self):
        from paper_rag.generation.service import build_rag_prompt

        prompt = build_rag_prompt(
            prompt_template="上下文:\n{context}\n问题:{question}",
            context="普通 context",
            question="GPT-3 做了什么？",
            verified_evidence_summary="【已校验证据】gpt3.pdf p7: 规模定律证据",
        )

        self.assertIn("【已校验证据】", prompt)
        self.assertIn("gpt3.pdf p7", prompt)
        self.assertLess(prompt.index("【已校验证据】"), prompt.index("普通 context"))

    def test_build_rag_prompt_without_verified_summary_keeps_existing_behavior(self):
        from paper_rag.generation.service import ANSWER_ORDER_INSTRUCTION, build_rag_prompt

        prompt = build_rag_prompt(
            prompt_template="上下文:{context}\n问题:{question}",
            context="普通 context",
            question="问题",
            history_text="历史\n",
        )

        self.assertEqual(
            prompt,
            "历史\n" + ANSWER_ORDER_INSTRUCTION + "\n上下文:普通 context\n问题:问题",
        )

    def test_generate_answer_passes_verified_summary_to_prompt(self):
        from paper_rag.generation.service import generate_answer

        class RecordingLlm:
            def __init__(self):
                self.prompt = ""

            def invoke(self, prompt):
                self.prompt = prompt
                return "答案"

        llm = RecordingLlm()

        answer = generate_answer(
            llm=llm,
            prompt_template="上下文:\n{context}\n问题:{question}",
            context="普通 context",
            question="问题",
            verified_evidence_summary="【已校验证据】gpt3.pdf p7",
        )

        self.assertEqual(answer, "答案")
        self.assertLess(llm.prompt.index("【已校验证据】"), llm.prompt.index("普通 context"))


if __name__ == "__main__":
    unittest.main()
