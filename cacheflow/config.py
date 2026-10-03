"""Configuration for CacheFlow's CARAS policy.

The vLLM custom scheduler constructor has no stable extension-config argument.
Environment variables are therefore the only integration mechanism used by the
adapter.  The pure policy accepts this dataclass directly for tests and simulations.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class SchedulingMode(str, Enum):
    """Supported ablations and the full policy."""

    FCFS = "fcfs"
    CACHE = "cache"
    RESIDUAL = "residual"
    AGING = "aging"
    CARAS = "caras"
    SLO = "slo"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {raw!r}")


@dataclass(frozen=True, slots=True)
class CARASConfig:
    """Policy parameters.

    ``prefill_tokens_per_ms`` should be calibrated for the model/GPU pair.  It
    converts residual tokens into a rough service-time estimate; it is not used
    by residual-only modes.
    """

    mode: SchedulingMode = SchedulingMode.CARAS
    candidate_window: int = 16
    max_wait_ms: float = 2_000.0
    aging_tokens_per_ms: float = 2.0
    cache_reuse_weight: float = 0.0
    min_cache_blocks: int = 1
    prefill_tokens_per_ms: float = 4.0
    attention_context_weight: float = 0.0
    attention_reference_tokens: int = 4_096
    ttft_slo_ms: float | None = None
    protect_preempted: bool = True
    metrics_path: Path | None = None
    metrics_interval: int = 100

    def __post_init__(self) -> None:
        for name in (
            "max_wait_ms",
            "aging_tokens_per_ms",
            "cache_reuse_weight",
            "prefill_tokens_per_ms",
            "attention_context_weight",
            "ttft_slo_ms",
        ):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.candidate_window < 1:
            raise ValueError("candidate_window must be >= 1")
        if self.max_wait_ms <= 0:
            raise ValueError("max_wait_ms must be > 0")
        if self.aging_tokens_per_ms < 0:
            raise ValueError("aging_tokens_per_ms must be >= 0")
        if self.cache_reuse_weight < 0:
            raise ValueError("cache_reuse_weight must be >= 0")
        if self.min_cache_blocks < 0:
            raise ValueError("min_cache_blocks must be >= 0")
        if self.prefill_tokens_per_ms <= 0:
            raise ValueError("prefill_tokens_per_ms must be > 0")
        if self.attention_context_weight < 0:
            raise ValueError("attention_context_weight must be >= 0")
        if self.attention_reference_tokens < 1:
            raise ValueError("attention_reference_tokens must be >= 1")
        if self.ttft_slo_ms is not None and self.ttft_slo_ms <= 0:
            raise ValueError("ttft_slo_ms must be > 0 when set")
        if self.metrics_interval < 1:
            raise ValueError("metrics_interval must be >= 1")

    @classmethod
    def from_env(cls, prefix: str = "CACHEFLOW_") -> CARASConfig:
        """Load configuration used by ``cacheflow.vllm_adapter.CARASScheduler``."""

        mode = SchedulingMode(os.getenv(f"{prefix}MODE", "caras").lower())
        slo_raw = os.getenv(f"{prefix}TTFT_SLO_MS")
        metrics_raw = os.getenv(f"{prefix}METRICS_PATH")
        return cls(
            mode=mode,
            candidate_window=int(os.getenv(f"{prefix}CANDIDATE_WINDOW", "16")),
            max_wait_ms=float(os.getenv(f"{prefix}MAX_WAIT_MS", "2000")),
            aging_tokens_per_ms=float(os.getenv(f"{prefix}AGING_TOKENS_PER_MS", "2.0")),
            cache_reuse_weight=float(os.getenv(f"{prefix}CACHE_REUSE_WEIGHT", "0.0")),
            min_cache_blocks=int(os.getenv(f"{prefix}MIN_CACHE_BLOCKS", "1")),
            prefill_tokens_per_ms=float(
                os.getenv(f"{prefix}PREFILL_TOKENS_PER_MS", "4.0")
            ),
            attention_context_weight=float(
                os.getenv(f"{prefix}ATTENTION_CONTEXT_WEIGHT", "0.0")
            ),
            attention_reference_tokens=int(
                os.getenv(f"{prefix}ATTENTION_REFERENCE_TOKENS", "4096")
            ),
            ttft_slo_ms=float(slo_raw) if slo_raw else None,
            protect_preempted=_env_bool(f"{prefix}PROTECT_PREEMPTED", True),
            metrics_path=Path(metrics_raw) if metrics_raw else None,
            metrics_interval=int(os.getenv(f"{prefix}METRICS_INTERVAL", "100")),
        )
