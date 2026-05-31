import unittest


class FakeClock:
    def __init__(self, values):
        self.values = list(values)

    def __call__(self):
        return self.values.pop(0)


class TraceTimerTest(unittest.TestCase):
    def test_trace_timer_records_stage_and_total_elapsed(self):
        from paper_rag.observability.trace import TraceTimer

        timer = TraceTimer(clock=FakeClock([10.0, 10.2, 10.4, 10.5, 11.0, 11.0]))

        rewrite_start = timer.start_stage()
        rewrite_elapsed = timer.elapsed_since(rewrite_start)
        retrieve_start = timer.start_stage()
        retrieve_elapsed = timer.elapsed_since(retrieve_start)

        self.assertAlmostEqual(rewrite_elapsed, 0.2)
        self.assertAlmostEqual(retrieve_elapsed, 0.5)
        elapsed = timer.elapsed_map(
            rewrite=rewrite_elapsed,
            retrieve=retrieve_elapsed,
            generate=0.0,
        )
        self.assertAlmostEqual(elapsed["rewrite"], 0.2)
        self.assertAlmostEqual(elapsed["retrieve"], 0.5)
        self.assertEqual(elapsed["generate"], 0.0)
        self.assertAlmostEqual(elapsed["total"], 1.0)


if __name__ == "__main__":
    unittest.main()
