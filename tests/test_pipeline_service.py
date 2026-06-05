import unittest


class FakeConversation:
    def __init__(self, history=None, rewritten="standalone question"):
        self.history = history or []
        self.rewritten = rewritten
        self.reformulate_calls = []

    def reformulate(self, question):
        self.reformulate_calls.append(question)
        return self.rewritten


class PipelineServiceTest(unittest.TestCase):
    def test_reformulate_question_can_force_rewrite_without_history(self):
        from paper_rag.pipeline.service import reformulate_question

        conversation = FakeConversation(history=[], rewritten="standalone question")

        result = reformulate_question(conversation, "follow up", require_history=False)

        self.assertEqual(result.standalone_question, "standalone question")
        self.assertTrue(result.rewritten)
        self.assertEqual(conversation.reformulate_calls, ["follow up"])

    def test_reformulate_question_skips_rewrite_when_history_required_but_empty(self):
        from paper_rag.pipeline.service import reformulate_question

        conversation = FakeConversation(history=[], rewritten="standalone question")

        result = reformulate_question(conversation, "first question", require_history=True)

        self.assertEqual(result.standalone_question, "first question")
        self.assertFalse(result.rewritten)
        self.assertEqual(conversation.reformulate_calls, [])

    def test_reformulate_question_reports_unchanged_question_as_not_rewritten(self):
        from paper_rag.pipeline.service import reformulate_question

        conversation = FakeConversation(history=[{"role": "user", "content": "history"}], rewritten="same question")

        result = reformulate_question(conversation, "same question", require_history=True)

        self.assertEqual(result.standalone_question, "same question")
        self.assertFalse(result.rewritten)
        self.assertEqual(conversation.reformulate_calls, ["same question"])


if __name__ == "__main__":
    unittest.main()
