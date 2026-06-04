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

    def test_doctor_command_forwards_remaining_args_to_diagnostics_cli(self):
        import main

        calls = []
        fake_diagnostics = types.SimpleNamespace(run_doctor_cli=lambda argv=None: calls.append(argv) or 7)

        with (
            patch.dict(sys.modules, {"paper_rag.config.diagnostics": fake_diagnostics}),
            patch.object(sys, "argv", ["main.py", "doctor", "--json"]),
        ):
            exit_code = main.main()

        self.assertEqual(calls, [["--json"]])
        self.assertEqual(exit_code, 7)


if __name__ == "__main__":
    unittest.main()
