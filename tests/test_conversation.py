import unittest

from conversation import ConversationManager


class EchoLLM:
    def invoke(self, prompt):
        return type("Response", (), {"content": "另一个预训练任务是什么？"})()


class ConversationManagerTest(unittest.TestCase):
    def test_reformulate_followup_keeps_source_from_history_when_llm_omits_it(self):
        manager = ConversationManager.__new__(ConversationManager)
        manager.llm = EchoLLM()
        manager.history = [
            {"role": "user", "content": "BERT 预训练时用了 Masked LM。"},
            {"role": "assistant", "content": "是的，BERT 使用 Masked LM 作为核心预训练任务之一。"},
        ]

        result = manager.reformulate("另一个预训练任务是什么？")

        self.assertEqual(result, "BERT 另一个预训练任务是什么？")


if __name__ == "__main__":
    unittest.main()
