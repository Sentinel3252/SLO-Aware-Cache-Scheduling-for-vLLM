from __future__ import annotations

import unittest

from cacheflow.config import CARASConfig, SchedulingMode
from cacheflow.request_queue import CARASRequestQueue
from tests.helpers import FakeFeatureProvider, FakeRequest


def queue(config: CARASConfig, policy: str = "fcfs") -> CARASRequestQueue:
    return CARASRequestQueue(policy, config, FakeFeatureProvider(config))


class RequestQueueTests(unittest.TestCase):
    def test_fcfs_equivalence_when_window_one(self) -> None:
        config = CARASConfig(mode=SchedulingMode.RESIDUAL, candidate_window=1)
        requests = [
            FakeRequest("long", 0, 1_000),
            FakeRequest("short", 1, 10),
        ]
        target = queue(config)
        for request in requests:
            target.add_request(request)
        target.prepare(now=2)
        self.assertEqual([request.request_id for request in target], ["long", "short"])

    def test_bounded_window_does_not_pull_from_tail(self) -> None:
        config = CARASConfig(
            mode=SchedulingMode.RESIDUAL,
            candidate_window=2,
            max_wait_ms=10_000,
        )
        target = queue(config)
        for request in (
            FakeRequest("a", 0, 100),
            FakeRequest("b", 1, 50),
            FakeRequest("tail", 2, 1),
        ):
            target.add_request(request)
        target.prepare(now=2.1)
        self.assertEqual([request.request_id for request in target], ["b", "a", "tail"])

    def test_priority_is_outer_order(self) -> None:
        config = CARASConfig(mode=SchedulingMode.RESIDUAL, candidate_window=8)
        target = queue(config, policy="priority")
        target.add_request(FakeRequest("low-short", 0, 1, priority=1))
        target.add_request(FakeRequest("high-long", 1, 10_000, priority=0))
        target.prepare(now=1.1)
        self.assertEqual(target.peek_request().request_id, "high-long")

    def test_preempted_fcfs_request_is_protected(self) -> None:
        config = CARASConfig(mode=SchedulingMode.RESIDUAL, candidate_window=8)
        target = queue(config)
        target.add_request(FakeRequest("short", 2, 1))
        preempted = FakeRequest("preempted", 0, 10_000, num_preemptions=1)
        target.prepend_request(preempted)
        target.prepare(now=2.1)
        self.assertIs(target.peek_request(), preempted)

    def test_cache_free_equal_cost_stays_fcfs(self) -> None:
        config = CARASConfig(
            mode=SchedulingMode.CARAS,
            candidate_window=8,
            aging_tokens_per_ms=0,
        )
        target = queue(config)
        for index in range(4):
            target.add_request(FakeRequest(str(index), float(index), 100))
        target.prepare(now=4)
        self.assertEqual(
            [request.request_id for request in target], ["0", "1", "2", "3"]
        )

    def test_remove_requests_covers_both_partitions(self) -> None:
        config = CARASConfig()
        target = queue(config)
        normal = FakeRequest("normal", 1, 10)
        preempted = FakeRequest("preempted", 0, 20, num_preemptions=1)
        target.add_request(normal)
        target.prepend_request(preempted)
        target.remove_requests([normal, preempted])
        self.assertFalse(target)

    def test_only_window_candidates_are_feature_probed(self) -> None:
        config = CARASConfig(mode=SchedulingMode.RESIDUAL, candidate_window=3)
        provider = FakeFeatureProvider(config)
        calls = 0

        def counting_provider(request, now):
            nonlocal calls
            calls += 1
            return provider(request, now)

        target = CARASRequestQueue("fcfs", config, counting_provider)
        for index in range(100):
            target.add_request(FakeRequest(str(index), float(index), 100 - index))
        result = target.prepare(now=101)
        self.assertEqual(calls, 3)
        self.assertEqual(result.candidates_scored, 3)

    def test_priority_readd_ignores_stale_heap_entry(self) -> None:
        config = CARASConfig(mode=SchedulingMode.RESIDUAL, candidate_window=4)
        target = queue(config, policy="priority")
        request = FakeRequest("same", 0, 100, priority=0)
        target.add_request(request)
        target.prepare(now=1)
        self.assertIs(target.pop_request(), request)
        request.num_preemptions += 1
        target.prepend_request(request)
        result = target.prepare(now=2)
        self.assertEqual(result.candidates_scored, 1)
        self.assertEqual([candidate.request_id for candidate in target], ["same"])


if __name__ == "__main__":
    unittest.main()
