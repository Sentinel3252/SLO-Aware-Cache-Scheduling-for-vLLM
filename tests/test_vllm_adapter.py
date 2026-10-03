from __future__ import annotations

import importlib
import os
import sys
import types
import unittest
from unittest.mock import patch


class FakeCoordinator:
    def find_longest_cache_hit(self, hashes, max_hit):
        return (), 0, 0


class FakeKVManager:
    coordinator = FakeCoordinator()

    def prefix_cache_lookup_enabled(self, request):
        return False


class FakeBaseScheduler:
    def __init__(self, *args, **kwargs):
        self.waiting = object()
        self.policy = types.SimpleNamespace(value="fcfs")
        self.kv_cache_manager = FakeKVManager()
        self.block_size = 16
        self.base_calls = []

    def schedule(self, *args, **kwargs):
        self.base_calls.append((args, kwargs))
        return {"unchanged_budget": kwargs.get("token_budget")}


class VLLMAdapterTests(unittest.TestCase):
    def test_adapter_delegates_budget_and_output_to_upstream(self) -> None:
        scheduler_module = types.ModuleType("vllm.v1.core.sched.scheduler")
        scheduler_module.Scheduler = FakeBaseScheduler
        fake_modules = {
            "vllm": types.ModuleType("vllm"),
            "vllm.v1": types.ModuleType("vllm.v1"),
            "vllm.v1.core": types.ModuleType("vllm.v1.core"),
            "vllm.v1.core.sched": types.ModuleType("vllm.v1.core.sched"),
            "vllm.v1.core.sched.scheduler": scheduler_module,
        }
        sys.modules.pop("cacheflow.vllm_adapter", None)
        with (
            patch.dict(sys.modules, fake_modules),
            patch.dict(os.environ, {"CACHEFLOW_MODE": "fcfs"}, clear=False),
        ):
            adapter = importlib.import_module("cacheflow.vllm_adapter")
            scheduler = adapter.CARASScheduler()
            output = scheduler.schedule(token_budget=17)
            self.assertEqual(output, {"unchanged_budget": 17})
            self.assertEqual(scheduler.base_calls, [((), {"token_budget": 17})])
            self.assertEqual(scheduler.cacheflow_metrics.snapshot().cycles, 1)
        sys.modules.pop("cacheflow.vllm_adapter", None)


if __name__ == "__main__":
    unittest.main()
