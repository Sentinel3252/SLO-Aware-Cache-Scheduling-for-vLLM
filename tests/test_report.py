from __future__ import annotations

import math
import unittest
from dataclasses import asdict

from benchmark.metrics_collector import RequestMetric
from benchmark.report import summarize


def row(
    request_id: str,
    *,
    ok: bool = True,
    start: float = 0,
    e2e: float = 100,
    ttft: float | None = 50,
    slo: float | None = 100,
) -> dict[str, object]:
    metric = RequestMetric(
        request_id, ok, 200 if ok else 500, start, ttft, None, e2e, 1 if ok else 0, slo
    )
    return {**asdict(metric), "slo_met": metric.slo_met}


class ReportTests(unittest.TestCase):
    def test_failures_stay_in_slo_denominator_and_duration(self) -> None:
        summary = summarize(
            [
                row("met", ttft=50),
                row("missed", ttft=150),
                row("failed", ok=False, start=200, e2e=800, ttft=None),
                row("untargeted", ttft=100, slo=None),
            ]
        )
        self.assertEqual(summary["requests"], 4)
        self.assertEqual(summary["failures"], 1)
        self.assertAlmostEqual(summary["slo_attainment"], 1 / 3)
        self.assertEqual(summary["slo_goodput_rps"], 1)
        self.assertEqual(summary["request_throughput_rps"], 3)
        self.assertEqual(summary["ttft_p50_ms"], 100)
        self.assertEqual(summary["ttft_p95_ms"], 145)
        self.assertEqual(summary["e2e_p95_ms"], 100)

    def test_missing_ttft_is_excluded_from_latency_but_not_slo_denominator(self):
        summary = summarize([row("no-text", ttft=None), row("text", ttft=20)])
        self.assertEqual(summary["ttft_p99_ms"], 20)
        self.assertEqual(summary["slo_attainment"], 0.5)

    def test_empty_and_all_failed_inputs(self) -> None:
        for rows in ([], [row("failed", ok=False, ttft=None)]):
            with self.subTest(rows=rows):
                summary = summarize(rows)
                self.assertEqual(summary["request_throughput_rps"], 0)
                self.assertEqual(summary["slo_goodput_rps"], 0)
                for key in (
                    "ttft_p50_ms",
                    "ttft_p95_ms",
                    "ttft_p99_ms",
                    "tpot_p95_ms",
                    "e2e_p95_ms",
                ):
                    self.assertTrue(math.isnan(summary[key]))
        self.assertTrue(math.isnan(summarize([])["slo_attainment"]))


if __name__ == "__main__":
    unittest.main()
