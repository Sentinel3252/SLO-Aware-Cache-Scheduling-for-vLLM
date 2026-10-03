from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cacheflow.metrics import PolicyMetrics
from cacheflow.models import ReorderResult


class MetricsTests(unittest.TestCase):
    def test_snapshot_is_isolated_from_caller_and_future_records(self) -> None:
        metrics = PolicyMetrics()
        result = ReorderResult(3, 1, 1000, 7)
        metrics.record(result)
        old = metrics.snapshot()
        old.reorder_histogram[0] = 99
        self.assertEqual(metrics.snapshot().reorder_histogram[0], 1)
        metrics.record(result)
        self.assertEqual(old.cycles, 1)
        self.assertEqual(old.reorder_histogram[0], 99)
        self.assertEqual(metrics.snapshot().cycles, 2)

    def test_interval_exports_cumulative_jsonl_and_creates_parents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "metrics.jsonl"
            metrics = PolicyMetrics(path, interval=2)
            result = ReorderResult(3, 1, 1000, 7)
            metrics.record(result)
            self.assertFalse(path.exists())
            for _ in range(3):
                metrics.record(result)
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([row["cycles"] for row in rows], [2, 4])
            self.assertEqual([row["candidates_scored"] for row in rows], [6, 12])
            self.assertEqual([row["overtakes"] for row in rows], [2, 4])
            self.assertTrue(all(row["mean_reorder_us"] == 1 for row in rows))
            self.assertTrue(all(row["p95_reorder_us_upper_bound"] == 1 for row in rows))
            self.assertEqual([sum(row["reorder_histogram"]) for row in rows], [2, 4])

    def test_no_path_does_not_export(self) -> None:
        metrics = PolicyMetrics(interval=1)
        with patch.object(metrics, "_append") as append:
            metrics.record(ReorderResult(1, 0, 1000, 1))
        append.assert_not_called()
        self.assertEqual(metrics.snapshot().cycles, 1)

    def test_latency_histogram_and_aggregates(self) -> None:
        metrics = PolicyMetrics()
        for elapsed_us in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
            metrics.record(
                ReorderResult(
                    candidates_scored=3,
                    overtakes=1,
                    elapsed_ns=elapsed_us * 1_000,
                    queue_length=7,
                )
            )
        snapshot = metrics.snapshot()
        self.assertEqual(snapshot.cycles, 10)
        self.assertEqual(snapshot.candidates_scored, 30)
        self.assertEqual(snapshot.overtakes, 10)
        self.assertEqual(snapshot.max_queue_length, 7)
        self.assertEqual(snapshot.mean_reorder_us, 102.3)
        self.assertEqual(snapshot.p95_reorder_us_upper_bound, 512)


if __name__ == "__main__":
    unittest.main()
