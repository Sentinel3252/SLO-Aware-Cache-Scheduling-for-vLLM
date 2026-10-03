from __future__ import annotations

import unittest

from cacheflow.config import CARASConfig, SchedulingMode
from cacheflow.cost_model import rank_key
from cacheflow.models import RequestFeature


def feature(
    request_id: str,
    *,
    remaining: int,
    wait_ms: float = 0,
    cached: int = 0,
    arrival: float = 0,
    slo_ms: float | None = None,
    estimated_ms: float | None = None,
) -> RequestFeature:
    return RequestFeature(
        request_id=request_id,
        arrival_time=arrival,
        priority=0,
        prompt_tokens=remaining + cached,
        cached_tokens=cached,
        remaining_prefill_tokens=remaining,
        waiting_time_ms=wait_ms,
        estimated_prefill_ms=estimated_ms
        if estimated_ms is not None
        else remaining / 4,
        required_kv_blocks=1,
        slo_ms=slo_ms,
    )


class CostModelTests(unittest.TestCase):
    def test_residual_prefers_less_remaining_work(self) -> None:
        config = CARASConfig(mode=SchedulingMode.RESIDUAL)
        short = feature("short", remaining=100)
        long = feature("long", remaining=1_000)
        self.assertLess(rank_key(short, config), rank_key(long, config))

    def test_aging_eventually_protects_long_request(self) -> None:
        config = CARASConfig(mode=SchedulingMode.AGING, max_wait_ms=500)
        old_long = feature("long", remaining=10_000, wait_ms=500, arrival=0)
        new_short = feature("short", remaining=10, wait_ms=0, arrival=1)
        self.assertLess(rank_key(old_long, config), rank_key(new_short, config))

    def test_slo_feasible_precedes_impossible(self) -> None:
        config = CARASConfig(mode=SchedulingMode.SLO, max_wait_ms=10_000)
        feasible = feature(
            "feasible", remaining=100, wait_ms=100, slo_ms=500, estimated_ms=25
        )
        impossible = feature(
            "impossible", remaining=10, wait_ms=600, slo_ms=500, estimated_ms=2.5
        )
        self.assertLess(rank_key(feasible, config), rank_key(impossible, config))

    def test_deterministic_tie_break(self) -> None:
        config = CARASConfig(mode=SchedulingMode.RESIDUAL)
        a = feature("a", remaining=10, arrival=1)
        b = feature("b", remaining=10, arrival=1)
        self.assertLess(rank_key(a, config), rank_key(b, config))


if __name__ == "__main__":
    unittest.main()
