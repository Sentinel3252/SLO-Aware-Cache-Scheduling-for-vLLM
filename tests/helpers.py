from __future__ import annotations

from dataclasses import dataclass, field

from cacheflow.config import CARASConfig
from cacheflow.models import RequestFeature


@dataclass(eq=False)
class FakeRequest:
    request_id: str
    arrival_time: float
    prompt_tokens: int
    cached_tokens: int = 0
    priority: int = 0
    num_preemptions: int = 0
    num_computed_tokens: int = 0
    num_tokens: int | None = None
    block_hashes: list[int] = field(default_factory=lambda: [1, 2, 3])

    def __post_init__(self) -> None:
        self.num_prompt_tokens = self.prompt_tokens
        if self.num_tokens is None:
            self.num_tokens = self.prompt_tokens


class FakeFeatureProvider:
    def __init__(self, config: CARASConfig, block_size: int = 16) -> None:
        self.config = config
        self.block_size = block_size

    def __call__(self, request: FakeRequest, now: float | None) -> RequestFeature:
        assert now is not None
        remaining = max(1, request.prompt_tokens - request.cached_tokens)
        waiting_ms = max(0.0, (now - request.arrival_time) * 1_000)
        return RequestFeature(
            request_id=request.request_id,
            arrival_time=request.arrival_time,
            priority=request.priority,
            prompt_tokens=request.prompt_tokens,
            cached_tokens=request.cached_tokens,
            remaining_prefill_tokens=remaining,
            waiting_time_ms=waiting_ms,
            estimated_prefill_ms=remaining / self.config.prefill_tokens_per_ms,
            required_kv_blocks=(remaining + self.block_size - 1) // self.block_size,
            slo_ms=self.config.ttft_slo_ms,
        )
