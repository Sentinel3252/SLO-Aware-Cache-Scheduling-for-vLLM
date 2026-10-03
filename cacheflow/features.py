"""Feature extraction, including a side-effect-free vLLM cache probe."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from cacheflow.config import CARASConfig
from cacheflow.models import RequestFeature


class VLLMFeatureProvider:
    """Extract read-only features from a vLLM V1 request.

    The regular ``KVCacheManager.get_computed_blocks`` call may emit events when
    full cache reporting is enabled.  This provider deliberately invokes the
    coordinator's hash lookup after the same eligibility guard.  The lookup does
    not allocate/touch blocks, update refcounts/LRU, or record hit metrics.

    This is necessarily an internal interface because vLLM does not yet expose
    the side-effect-free scheduler feature proposed in RFC #51608.
    """

    def __init__(
        self,
        kv_cache_manager: Any,
        block_size: int,
        config: CARASConfig,
        clock: Callable[[], float],
    ) -> None:
        self.kv_cache_manager = kv_cache_manager
        self.block_size = block_size
        self.config = config
        self.clock = clock

    def __call__(self, request: Any, now: float | None = None) -> RequestFeature:
        current = self.clock() if now is None else now
        prompt_tokens = int(request.num_prompt_tokens)
        cached_tokens = self._cached_tokens(request)
        remaining = max(1, prompt_tokens - cached_tokens)
        waiting_ms = max(0.0, (current - float(request.arrival_time)) * 1_000)
        # Linear residual tokens are a useful baseline but understate the cost
        # of new queries attending over a long cached prefix. This configurable
        # term approximates c*r + r^2/2 attention work in equivalent tokens.
        context = cached_tokens + remaining / 2
        equivalent_tokens = remaining * (
            1
            + self.config.attention_context_weight
            * context
            / self.config.attention_reference_tokens
        )
        per_request_slo = getattr(request, "cacheflow_ttft_slo_ms", None)
        slo_ms = (
            float(per_request_slo)
            if per_request_slo is not None
            else self.config.ttft_slo_ms
        )
        return RequestFeature(
            request_id=str(request.request_id),
            arrival_time=float(request.arrival_time),
            priority=int(getattr(request, "priority", 0)),
            prompt_tokens=prompt_tokens,
            cached_tokens=cached_tokens,
            remaining_prefill_tokens=remaining,
            waiting_time_ms=waiting_ms,
            estimated_prefill_ms=(
                equivalent_tokens / self.config.prefill_tokens_per_ms
            ),
            required_kv_blocks=math.ceil(remaining / self.block_size),
            slo_ms=slo_ms,
        )

    def _cached_tokens(self, request: Any) -> int:
        manager = self.kv_cache_manager
        if int(getattr(request, "num_computed_tokens", 0)) != 0:
            return 0
        lookup_enabled = getattr(manager, "prefix_cache_lookup_enabled", None)
        if lookup_enabled is not None:
            enabled = bool(lookup_enabled(request))
        else:
            # vLLM v0.23.x keeps this guard inline in get_computed_blocks;
            # newer main exposes prefix_cache_lookup_enabled().
            enabled = bool(getattr(manager, "enable_caching", False)) and not bool(
                getattr(request, "skip_reading_prefix_cache", False)
            )
        if not enabled:
            return 0
        max_hit = max(0, int(request.num_tokens) - 1)
        lookup_result = manager.coordinator.find_longest_cache_hit(
            request.block_hashes, max_hit
        )
        if len(lookup_result) == 2:  # vLLM v0.23.x
            _blocks, hit_tokens = lookup_result
        elif len(lookup_result) == 3:  # current main / hybrid shared-boundary API
            _blocks, hit_tokens, _uncached = lookup_result
        else:  # fail loudly on an upstream contract change
            raise RuntimeError(
                "Unsupported vLLM find_longest_cache_hit return shape: "
                f"{len(lookup_result)}"
            )
        hit_tokens = int(hit_tokens)
        if hit_tokens < self.config.min_cache_blocks * self.block_size:
            return 0
        return min(hit_tokens, max(0, int(request.num_prompt_tokens) - 1))
