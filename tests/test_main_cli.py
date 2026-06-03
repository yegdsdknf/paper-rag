import sys
import types
import unittest
from unittest.mock import patch


class MainCliTest(unittest.TestCase):
    def test_build_command_forwards_remaining_args_to_build_knowledge(self):
        import main

        calls = []
        fake_build_knowledge = types.SimpleNamespace(main=lambda argv=None: calls.append(argv))

        with (
            patch.dict(sys.modules, {"build_knowledge": fake_build_knowledge}),
            patch.object(
                sys,
                "argv",
                ["main.py", "build", "--experiment", "section-aware", "--rebuild"],
            ),
        ):
            main.main()

        self.assertEqual(calls, [["--experiment", "section-aware", "--rebuild"]])

    def test_query_command_forwards_remaining_args_to_query(self):
        import main

        calls = []
        fake_query = types.SimpleNamespace(main=lambda argv=None: calls.append(argv))

        with (
            patch.dict(sys.modules, {"query": fake_query}),
            patch.object(sys, "argv", ["main.py", "query", "--agent"]),
        ):
            main.main()

        self.assertEqual(calls, [["--agent"]])

    def test_query_flags_without_command_default_to_query(self):
        import main

        calls = []
        fake_query = types.SimpleNamespace(main=lambda argv=None: calls.append(argv))

        with patch.dict(sys.modules, {"query": fake_query}):
            main.main(["--agent"])

        self.assertEqual(calls, [["--agent"]])

    def test_top_level_help_is_not_forwarded_to_query(self):
        import main

        calls = []
        fake_query = types.SimpleNamespace(main=lambda argv=None: calls.append(argv))
        printed = []

        with (
            patch.dict(sys.modules, {"query": fake_query}),
            patch("builtins.print", lambda *args, **kwargs: printed.append(" ".join(str(arg) for arg in args))),
        ):
            main.main(["--help"])

        self.assertEqual(calls, [])
        self.assertTrue(any("python main.py build" in line for line in printed))


if __name__ == "__main__":
    unittest.main()
