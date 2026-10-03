from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cacheflow.tools.output_equivalence import capture, compare


class OutputEquivalenceTests(unittest.TestCase):
    def test_empty_or_nonmapping_capture_files_fail(self) -> None:
        valid = {"r": {"text": "answer"}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            for left, right in (
                ({}, {}),
                ({}, valid),
                (valid, {}),
                ([], valid),
                (valid, []),
            ):
                with self.subTest(left=left, right=right):
                    baseline.write_text(json.dumps(left), encoding="utf-8")
                    candidate.write_text(json.dumps(right), encoding="utf-8")
                    output = io.StringIO()
                    with (
                        contextlib.redirect_stdout(output),
                        self.assertRaisesRegex(
                            SystemExit, "nonempty request-id mapping"
                        ),
                    ):
                        compare(
                            argparse.Namespace(baseline=baseline, candidate=candidate)
                        )
                    self.assertNotIn("OK", output.getvalue())

    def test_capture_rejects_empty_trace_and_invalid_limit_before_network(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "empty.jsonl"
            trace.write_text("\n", encoding="utf-8")
            args = argparse.Namespace(
                trace=trace, output=root / "capture.json", limit=32
            )
            for limit in (0, -1, 32):
                with self.subTest(limit=limit):
                    args.limit = limit
                    with (
                        patch(
                            "cacheflow.tools.output_equivalence._complete"
                        ) as complete,
                        self.assertRaises(SystemExit),
                    ):
                        capture(args)
                    complete.assert_not_called()
                    self.assertFalse(args.output.exists())

    def test_request_id_difference_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline, candidate = root / "baseline.json", root / "candidate.json"
            baseline.write_text(json.dumps({"a": {"text": "same"}}), encoding="utf-8")
            candidate.write_text(json.dumps({"b": {"text": "same"}}), encoding="utf-8")
            with (
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaises(SystemExit) as raised,
            ):
                compare(argparse.Namespace(baseline=baseline, candidate=candidate))
            self.assertEqual(raised.exception.code, 1)

    def test_equal_capture_files_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            payload = {"r": {"text": "answer", "finish_reason": "length"}}
            baseline.write_text(json.dumps(payload), encoding="utf-8")
            candidate.write_text(json.dumps(payload), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                compare(argparse.Namespace(baseline=baseline, candidate=candidate))

    def test_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            baseline.write_text(json.dumps({"r": {"text": "a"}}), encoding="utf-8")
            candidate.write_text(json.dumps({"r": {"text": "b"}}), encoding="utf-8")
            with (
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                compare(argparse.Namespace(baseline=baseline, candidate=candidate))


if __name__ == "__main__":
    unittest.main()
