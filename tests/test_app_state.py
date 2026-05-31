import unittest

from app_state import clear_conversation_state


class AppStateTest(unittest.TestCase):
    def test_clear_conversation_state_removes_feedback_payload(self):
        state = {
            "messages": [{"role": "assistant", "content": "answer"}],
            "conversation": object(),
            "last_feedback_payload": {"question": "old question"},
            "feedback_note": "old note",
            "selected_llm_model": "qwen2.5:3b",
        }

        clear_conversation_state(state)

        self.assertEqual(state["messages"], [])
        self.assertIsNone(state["conversation"])
        self.assertIsNone(state["last_feedback_payload"])
        self.assertNotIn("feedback_note", state)
        self.assertEqual(state["selected_llm_model"], "qwen2.5:3b")


if __name__ == "__main__":
    unittest.main()
