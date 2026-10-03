from __future__ import annotations

import unittest
from unittest.mock import patch

from cacheflow.config import CARASConfig
from cacheflow.features import VLLMFeatureProvider
from tests.helpers import FakeRequest


class FakeCoordinator:
    def __init__(self, hit_tokens: int) -> None:
        self.hit_tokens = hit_tokens
        self.calls = 0

    def find_longest_cache_hit(self, hashes, max_hit):
        self.calls += 1
        return ("read-only-block-view",), min(self.hit_tokens, max_hit), 0


class FakeManager:
    def __init__(self, hit_tokens: int) -> None:
        self.coordinator = FakeCoordinator(hit_tokens)
        self.events: list[object] = []
        self.refcount = 7

    def prefix_cache_lookup_enabled(self, request) -> bool:
        return True

    def get_computed_blocks(self, request):
        raise AssertionError("side-effecting public probe must not be called")


class StableCoordinator(FakeCoordinator):
    def find_longest_cache_hit(self, hashes, max_hit):
        self.calls += 1
        return ("read-only-block-view",), min(self.hit_tokens, max_hit)


class StableManager:
    def __init__(self, hit_tokens: int) -> None:
        self.coordinator = StableCoordinator(hit_tokens)
        self.enable_caching = True


class FeatureTests(unittest.TestCase):
    def test_disabled_lookup_guards_do_not_probe(self) -> None:
        manager = FakeManager(hit_tokens=32)
        provider = VLLMFeatureProvider(manager, 16, CARASConfig(), clock=lambda: 1.0)
        with patch.object(manager, "prefix_cache_lookup_enabled", return_value=False):
            self.assertEqual(provider(FakeRequest("disabled", 0, 100)).cached_tokens, 0)
        self.assertEqual(manager.coordinator.calls, 0)
        for enable_caching, skip in ((False, False), (True, True)):
            with self.subTest(enable_caching=enable_caching, skip=skip):
                stable = StableManager(hit_tokens=32)
                stable.enable_caching = enable_caching
                request = FakeRequest("stable", 0, 100)
                request.skip_reading_prefix_cache = skip
                provider = VLLMFeatureProvider(
                    stable, 16, CARASConfig(), clock=lambda: 1.0
                )
                self.assertEqual(provider(request).cached_tokens, 0)
                self.assertEqual(stable.coordinator.calls, 0)

    def test_unknown_lookup_shapes_fail_loudly(self) -> None:
        manager = FakeManager(hit_tokens=32)
        provider = VLLMFeatureProvider(manager, 16, CARASConfig(), clock=lambda: 1.0)
        for result in (((),), ((), 32, 0, "unexpected")):
            with (
                self.subTest(result=result),
                patch.object(
                    manager.coordinator, "find_longest_cache_hit", return_value=result
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, f"return shape: {len(result)}"
                ):
                    provider(FakeRequest("r", 0, 100))

    def test_cache_probe_has_no_adapter_side_effects(self) -> None:
        manager = FakeManager(hit_tokens=64)
        provider = VLLMFeatureProvider(
            manager,
            block_size=16,
            config=CARASConfig(prefill_tokens_per_ms=2),
            clock=lambda: 2.0,
        )
        request = FakeRequest("r", 1.0, 100)
        result = provider(request)
        self.assertEqual(result.cached_tokens, 64)
        self.assertEqual(result.remaining_prefill_tokens, 36)
        self.assertEqual(result.required_kv_blocks, 3)
        self.assertEqual(manager.coordinator.calls, 1)
        self.assertEqual(manager.events, [])
        self.assertEqual(manager.refcount, 7)

    def test_precomputed_request_is_not_probed(self) -> None:
        manager = FakeManager(hit_tokens=64)
        provider = VLLMFeatureProvider(manager, 16, CARASConfig(), clock=lambda: 2.0)
        request = FakeRequest("r", 1.0, 100, num_computed_tokens=32)
        self.assertEqual(provider(request).cached_tokens, 0)
        self.assertEqual(manager.coordinator.calls, 0)

    def test_v023_two_value_lookup_and_inline_guard(self) -> None:
        manager = StableManager(hit_tokens=32)
        provider = VLLMFeatureProvider(manager, 16, CARASConfig(), clock=lambda: 1.0)
        request = FakeRequest("stable", 0, 100)
        request.skip_reading_prefix_cache = False
        self.assertEqual(provider(request).cached_tokens, 32)
        self.assertEqual(manager.coordinator.calls, 1)

    def test_minimum_cache_block_threshold(self) -> None:
        manager = FakeManager(hit_tokens=16)
        provider = VLLMFeatureProvider(
            manager, 16, CARASConfig(min_cache_blocks=2), clock=lambda: 1.0
        )
        self.assertEqual(provider(FakeRequest("r", 0, 100)).cached_tokens, 0)

    def test_context_attention_term_distinguishes_equal_residual(self) -> None:
        warm_manager = FakeManager(hit_tokens=80)
        cold_manager = FakeManager(hit_tokens=0)
        config = CARASConfig(
            prefill_tokens_per_ms=1,
            attention_context_weight=1,
            attention_reference_tokens=100,
        )
        warm = VLLMFeatureProvider(warm_manager, 16, config, clock=lambda: 1.0)(
            FakeRequest("warm", 0, 100)
        )
        cold = VLLMFeatureProvider(cold_manager, 16, config, clock=lambda: 1.0)(
            FakeRequest("cold", 0, 20)
        )
        self.assertEqual(warm.remaining_prefill_tokens, cold.remaining_prefill_tokens)
        self.assertGreater(warm.estimated_prefill_ms, cold.estimated_prefill_ms)


if __name__ == "__main__":
    unittest.main()
