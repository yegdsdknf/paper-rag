import sys
import types
import unittest
from io import StringIO
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

    def test_build_command_formats_runtime_error_and_returns_failure_code(self):
        import main

        def fail_build(_argv=None):
            raise RuntimeError("collection is empty")

        fake_build_knowledge = types.SimpleNamespace(main=fail_build)
        stderr = StringIO()

        with (
            patch.dict(sys.modules, {"build_knowledge": fake_build_knowledge}),
            patch.object(sys, "argv", ["main.py", "build"]),
            patch("sys.stderr", stderr),
        ):
            exit_code = main.main()

        output = stderr.getvalue()
        self.assertEqual(exit_code, 1)
        self.assertIn("向量库未构建或为空", output)
        self.assertIn("python main.py build", output)
        self.assertIn("python main.py doctor", output)
        self.assertIn("RuntimeError: collection is empty", output)


if __name__ == "__main__":
    unittest.main()
