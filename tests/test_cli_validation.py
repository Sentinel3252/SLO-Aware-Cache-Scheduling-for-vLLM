from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark.client import run
from benchmark.schema import read_trace
from benchmark.workload_generator import main as generate


class CLIValidationTests(unittest.TestCase):
    def test_invalid_generator_inputs_do_not_write_output(self) -> None:
        cases = [
            ["--workload", "multiturn", "--sessions", "0"],
            ["--workload", "multiturn", "--sessions", "-1"],
            ["--workload", "multiturn", "--turns", "0"],
            ["--workload", "multiturn", "--turns", "-1"],
            ["--workload", "mixed", "--mixed-lengths", "0"],
            ["--workload", "mixed", "--mixed-lengths", "-2"],
            ["--workload", "cold", "--count", "0"],
        ]
        cases.extend(
            ["--workload", "rag", flag, raw]
            for flag in ("--qps", "--slo-ms")
            for raw in ("0", "-1", "nan", "inf")
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "trace.jsonl"
            for flags in cases:
                with self.subTest(flags=flags):
                    error = io.StringIO()
                    with (
                        contextlib.redirect_stderr(error),
                        self.assertRaises(SystemExit) as raised,
                    ):
                        generate([*flags, "--output", str(output)])
                    self.assertEqual(raised.exception.code, 2)
                    self.assertIn("must be", error.getvalue())
                    self.assertFalse(output.exists())

    def test_valid_multiturn_uses_dimensions_and_mixed_is_seeded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "trace.jsonl"
            with contextlib.redirect_stdout(io.StringIO()):
                generate(
                    [
                        "--workload",
                        "multiturn",
                        "--sessions",
                        "2",
                        "--turns",
                        "3",
                        "--count",
                        "0",
                        "--output",
                        str(output),
                    ]
                )
                self.assertEqual(len(read_trace(output)), 6)
                flags = [
                    "--workload",
                    "mixed",
                    "--count",
                    "4",
                    "--mixed-lengths",
                    "8",
                    "16",
                    "--seed",
                    "7",
                    "--output",
                    str(output),
                ]
                generate(flags)
                first = output.read_bytes()
                self.assertTrue(
                    all(r.prompt_tokens_estimate in {8, 16} for r in read_trace(output))
                )
                generate(flags)
                self.assertEqual(first, output.read_bytes())

    def test_invalid_benchmark_limits_fail_before_imports_or_io(self) -> None:
        for concurrency, timeout in (
            (0, 600),
            (-1, 600),
            (1, 0),
            (1, -1),
            (1, float("nan")),
            (1, float("inf")),
        ):
            with self.subTest(concurrency=concurrency, timeout=timeout):
                args = argparse.Namespace(max_concurrency=concurrency, timeout=timeout)
                with (
                    patch("benchmark.client.read_trace") as read,
                    self.assertRaises(SystemExit),
                ):
                    asyncio.run(run(args))
                read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
