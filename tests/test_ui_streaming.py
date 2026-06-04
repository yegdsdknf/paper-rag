import unittest


class TokenStreamBufferTest(unittest.TestCase):
    def test_flushes_when_chunk_threshold_is_reached(self):
        from paper_rag.ui.streaming import TokenStreamBuffer

        buffer = TokenStreamBuffer(max_chunks=3, max_interval_sec=10.0)

        self.assertEqual(buffer.append("a", now=0.0), "")
        self.assertEqual(buffer.append("b", now=0.1), "")
        self.assertEqual(buffer.append("c", now=0.2), "abc")
        self.assertEqual(buffer.flush(), "")

    def test_flushes_when_time_threshold_is_reached(self):
        from paper_rag.ui.streaming import TokenStreamBuffer

        buffer = TokenStreamBuffer(max_chunks=8, max_interval_sec=0.08)

        self.assertEqual(buffer.append("a", now=1.0), "")
        self.assertEqual(buffer.append("b", now=1.09), "ab")

    def test_explicit_flush_returns_tail_once(self):
        from paper_rag.ui.streaming import TokenStreamBuffer

        buffer = TokenStreamBuffer(max_chunks=8, max_interval_sec=10.0)
        buffer.append("tail", now=0.0)

        self.assertEqual(buffer.flush(), "tail")
        self.assertEqual(buffer.flush(), "")

    def test_ignores_empty_chunks(self):
        from paper_rag.ui.streaming import TokenStreamBuffer

        buffer = TokenStreamBuffer(max_chunks=1, max_interval_sec=0.0)

        self.assertEqual(buffer.append("", now=0.0), "")
        self.assertEqual(buffer.flush(), "")


if __name__ == "__main__":
    unittest.main()
