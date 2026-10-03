from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from cacheflow.config import CARASConfig, SchedulingMode
from cacheflow.models import RequestFeature
from cacheflow.request_queue import CARASRequestQueue


@dataclass(eq=False)
class ContractRequest:
    request_id: str
    arrival_time: float
    priority: int = 0
    num_preemptions: int = 0

    def __lt__(self, other: ContractRequest) -> bool:
        return (self.priority, self.arrival_time, self.request_id) < (
            other.priority,
            other.arrival_time,
            other.request_id,
        )


def features(request: ContractRequest, now: float | None) -> RequestFeature:
    assert now is not None
    return RequestFeature(
        request_id=request.request_id,
        arrival_time=request.arrival_time,
        priority=request.priority,
        prompt_tokens=1,
        cached_tokens=0,
        remaining_prefill_tokens=1,
        waiting_time_ms=max(0, (now - request.arrival_time) * 1_000),
        estimated_prefill_ms=1,
        required_kv_blocks=1,
    )


@unittest.skipUnless(
    os.getenv("VLLM_SOURCE_TREE"), "set VLLM_SOURCE_TREE for upstream contract test"
)
class UpstreamContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        source = (
            Path(os.environ["VLLM_SOURCE_TREE"]) / "vllm/v1/core/sched/request_queue.py"
        )
        request_module = types.ModuleType("vllm.v1.request")
        request_module.Request = ContractRequest
        spec = importlib.util.spec_from_file_location("_upstream_request_queue", source)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"vllm.v1.request": request_module}):
            spec.loader.exec_module(module)
        cls.upstream = module

    def test_fcfs_contract_matches_upstream(self) -> None:
        config = CARASConfig(
            mode=SchedulingMode.FCFS,
            candidate_window=8,
            max_wait_ms=1_000_000,
        )
        ours = CARASRequestQueue("fcfs", config, features)
        theirs = self.upstream.FCFSRequestQueue()
        a = ContractRequest("a", 0)
        b = ContractRequest("b", 1)
        p = ContractRequest("p", 2, num_preemptions=1)
        for target in (ours, theirs):
            target.add_request(a)
            target.add_request(b)
            target.prepend_request(p)
        ours.prepare(now=3)
        self.assertEqual(
            [request.request_id for request in ours],
            [request.request_id for request in theirs],
        )
        self.assertEqual(ours.pop_request().request_id, theirs.pop_request().request_id)

    def test_priority_contract_matches_upstream(self) -> None:
        config = CARASConfig(
            mode=SchedulingMode.FCFS,
            candidate_window=8,
            max_wait_ms=1_000_000,
        )
        ours = CARASRequestQueue("priority", config, features)
        theirs = self.upstream.PriorityRequestQueue()
        requests = [
            ContractRequest("low", 0, priority=3),
            ContractRequest("high-new", 2, priority=0),
            ContractRequest("high-old", 1, priority=0),
        ]
        for request in requests:
            ours.add_request(request)
            theirs.add_request(request)
        ours.prepare(now=3)
        self.assertEqual(
            [request.request_id for request in ours],
            [request.request_id for request in theirs],
        )


if __name__ == "__main__":
    unittest.main()
