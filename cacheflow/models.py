"""Policy-facing immutable data structures."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RequestFeature:
    request_id: str
    arrival_time: float
    priority: int
    prompt_tokens: int
    cached_tokens: int
    remaining_prefill_tokens: int
    waiting_time_ms: float
    estimated_prefill_ms: float
    required_kv_blocks: int
    slo_ms: float | None = None

    @property
    def cache_ratio(self) -> float:
        return self.cached_tokens / self.prompt_tokens if self.prompt_tokens else 0.0

    @property
    def slack_ms(self) -> float | None:
        if self.slo_ms is None:
            return None
        return self.slo_ms - self.waiting_time_ms - self.estimated_prefill_ms


@dataclass(frozen=True, slots=True)
class ReorderResult:
    candidates_scored: int
    overtakes: int
    elapsed_ns: int
    queue_length: int
