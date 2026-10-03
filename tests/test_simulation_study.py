from __future__ import annotations

import math
import unittest

from cacheflow.config import SchedulingMode
from cacheflow.tools.run_simulation_study import _mean_ci95, render_report, run_study


class SimulationStudyTests(unittest.TestCase):
    def test_mean_ci95(self) -> None:
        mean, ci = _mean_ci95([1.0, 2.0, 3.0])
        self.assertEqual(mean, 2.0)
        self.assertGreater(ci, 0)
        self.assertTrue(math.isnan(_mean_ci95([1.0])[1]))

    def test_study_covers_cells_and_renders_warning(self) -> None:
        rows = run_study(
            workloads=["rag"],
            loads=[100.0],
            seeds=[1, 2],
            modes=[SchedulingMode.FCFS, SchedulingMode.CARAS],
            count=8,
            slo_ms=1_000,
            candidate_windows=[4],
            prefill_tokens_per_ms=4.0,
            attention_context_weight=0.0,
        )
        self.assertEqual(len(rows), 4)
        report = render_report(rows)
        self.assertIn("not vLLM/GPU", report)
        self.assertIn("| rag | 100 | 4 | caras |", report)


if __name__ == "__main__":
    unittest.main()
