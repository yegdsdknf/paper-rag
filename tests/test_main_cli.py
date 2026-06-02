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


if __name__ == "__main__":
    unittest.main()
