from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

from benchmark.schema import TraceRequest, read_trace, write_trace


class TraceSchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.trace = Path(directory.name) / "input.jsonl"
        self.record = {
            "request_id": "request-1",
            "arrival_ms": 0,
            "prompt": "你好，CacheFlow!",
            "prompt_tokens_estimate": 4,
        }

    def test_keeps_defaults_unicode_order_and_blank_lines_when_round_tripping(self):
        row = {
            **self.record,
            "request_id": "request-2",
            "arrival_ms": 9.5,
            "prefix_group": "共享前缀",
            "ttft_slo_ms": 20,
            "max_tokens": 8,
        }
        self.trace.write_text(
            json.dumps(row, ensure_ascii=False)
            + "\r\n \t\r\n"
            + json.dumps(self.record, ensure_ascii=False)
            + "\r\n",
            encoding="utf-8",
            newline="",
        )
        requests = read_trace(self.trace)
        self.assertEqual(
            requests,
            [
                TraceRequest(**row),
                TraceRequest("request-1", 0, "你好，CacheFlow!", 4),
            ],
        )
        write_trace(self.trace, requests)
        self.assertEqual(read_trace(self.trace), requests)

    def test_rejects_duplicate_ids_at_their_physical_line_number(self):
        first = json.dumps(self.record)
        duplicate = json.dumps({**self.record, "prompt": "Other request"})
        self.trace.write_text(f"\n{first}\n\n{duplicate}\n", encoding="utf-8")

        location = re.escape(str(self.trace))
        with self.assertRaisesRegex(
            ValueError, rf"{location}:4: .*request-1.*first defined on line 2"
        ):
            read_trace(self.trace)

    def test_rejects_invalid_types_ranges_and_nonfinite_numbers(self):
        invalid_fields = {
            "request_id": ("", " \t", 1),
            "prompt": (None, ["text"]),
            "prompt_tokens_estimate": (True, 0, -1, 2.0, "2"),
            "max_tokens": (False, 0, -2, 2.0),
            "arrival_ms": (True, -1, "2", float("nan"), float("inf"), 10**400),
            "ttft_slo_ms": (False, -1, 0, "50", float("-inf"), float("nan")),
            "prefix_group": (2, {}),
        }
        for field, values in invalid_fields.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    record = {**self.record, field: value}
                    self.trace.write_text(
                        "\n" + json.dumps(record) + "\n", encoding="utf-8"
                    )
                    location = re.escape(str(self.trace))
                    with self.assertRaisesRegex(
                        ValueError, rf"{location}:2: .*{field}"
                    ):
                        read_trace(self.trace)

    def test_locates_invalid_json_missing_unknown_and_non_object_records(self):
        missing = {key: val for key, val in self.record.items() if key != "prompt"}
        unknown = {**self.record, "priority_class": "normal"}
        for record, message in (
            (b"{invalid", "(?:JSON|Expecting)"),
            (json.dumps(missing).encode(), "prompt"),
            (json.dumps(unknown).encode(), "priority_class"),
            (b"[]", "JSON object"),
            (b'{"prompt": "\xff"}', "utf-8"),
        ):
            with self.subTest(record=record):
                self.trace.write_bytes(b"\n" + record + b"\n")
                location = re.escape(str(self.trace))
                with self.assertRaisesRegex(ValueError, rf"{location}:2: .*{message}"):
                    read_trace(self.trace)
