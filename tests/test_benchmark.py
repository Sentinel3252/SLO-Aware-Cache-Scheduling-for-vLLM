from __future__ import annotations

import math
import random
import unittest

from benchmark.metrics_collector import percentile
from benchmark.simulator import simulate
from benchmark.workloads import cache_free, poisson_arrivals, rag
from cacheflow.config import CARASConfig, SchedulingMode


class BenchmarkTests(unittest.TestCase):
    def test_poisson_rejects_nonpositive_and_nonfinite_rates(self) -> None:
        for qps in (0, -1, math.nan, math.inf, -math.inf):
            with self.subTest(qps=qps), self.assertRaisesRegex(ValueError, "qps"):
                poisson_arrivals(1, qps, random.Random(7))

    def test_rag_has_distinct_stable_prefixes_beyond_twenty_groups(self) -> None:
        one = rag(500, qps=1, seed=7, groups=40, prefix_tokens=32, suffix_tokens=8)
        self.assertEqual(
            one, rag(500, qps=1, seed=7, groups=40, prefix_tokens=32, suffix_tokens=8)
        )
        prefixes: dict[str, str] = {}
        for request in one:
            prefix = request.prompt.split("\nQuestion ")[0]
            if request.prefix_group in prefixes:
                self.assertEqual(prefixes[request.prefix_group], prefix)
            else:
                prefixes[request.prefix_group] = prefix
            self.assertEqual(len(prefix.split()), 32)
            self.assertEqual(request.prompt_tokens_estimate, 43)
        self.assertEqual(len(prefixes), 40)
        self.assertEqual(len(set(prefixes.values())), 40)

    def test_rag_rejects_invalid_group_or_prefix_dimensions(self) -> None:
        for kwargs in ({"groups": 0}, {"groups": -1}, {"prefix_tokens": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                rag(1, qps=1, seed=7, **kwargs)

    def test_poisson_arrivals_are_monotonic_and_seeded(self) -> None:
        one = poisson_arrivals(10, 2.0, random.Random(9))
        two = poisson_arrivals(10, 2.0, random.Random(9))
        self.assertEqual(one, two)
        self.assertTrue(all(a < b for a, b in zip(one, one[1:], strict=False)))

    def test_cache_free_nonce_changes_first_prefix(self) -> None:
        requests = cache_free(3, qps=1, seed=7, prompt_tokens=20)
        first_words = [request.prompt.split()[0] for request in requests]
        self.assertEqual(len(set(first_words)), 3)

    def test_rag_simulator_observes_cache_reuse(self) -> None:
        requests = rag(
            8,
            qps=100,
            seed=3,
            prefix_tokens=64,
            suffix_tokens=8,
            groups=1,
        )
        rows = simulate(
            requests,
            CARASConfig(
                mode=SchedulingMode.CARAS,
                candidate_window=4,
                prefill_tokens_per_ms=1,
            ),
        )
        self.assertEqual(rows[0]["cached_tokens"], 0)
        self.assertTrue(any(row["cached_tokens"] > 0 for row in rows[1:]))

    def test_percentile_interpolates(self) -> None:
        self.assertEqual(percentile([0, 10], 0.5), 5)


if __name__ == "__main__":
    unittest.main()
