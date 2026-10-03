"""Deterministic CARAS ranking functions.

Keys sort in ascending order.  Priority is deliberately not part of this module:
the queue makes it an outer key so CacheFlow never changes vLLM priority classes.
"""

from __future__ import annotations

from typing import Any

from cacheflow.config import CARASConfig, SchedulingMode
from cacheflow.models import RequestFeature


def _tie(feature: RequestFeature) -> tuple[float, str]:
    return feature.arrival_time, feature.request_id


def rank_key(feature: RequestFeature, config: CARASConfig) -> tuple[Any, ...]:
    """Return a stable ascending rank key for one candidate.

    Starvation protection is an outer class.  The SLO mode uses a
    feasibility-first least-laxity heuristic: requests that can still meet the
    target precede already-impossible requests; aging eventually overrides this
    separation for visible candidates through ``max_wait_ms``. This is not a
    wall-clock wait guarantee. It avoids the singularity and
    post-deadline collapse of ``1 / max(slack, epsilon)``.
    """

    tie = _tie(feature)
    if feature.waiting_time_ms >= config.max_wait_ms:
        return (0, *tie)

    mode = config.mode
    if mode is SchedulingMode.FCFS:
        return (1, *tie)

    if mode is SchedulingMode.CACHE:
        return (1, -feature.cached_tokens, *tie)

    if mode is SchedulingMode.RESIDUAL:
        return (1, feature.remaining_prefill_tokens, *tie)

    aged_residual = max(
        0.0,
        feature.remaining_prefill_tokens
        - config.aging_tokens_per_ms * feature.waiting_time_ms,
    )
    if mode is SchedulingMode.AGING:
        return (1, aged_residual, *tie)

    estimated_work_tokens = feature.estimated_prefill_ms * config.prefill_tokens_per_ms
    effective_work = max(
        0.0,
        estimated_work_tokens
        - config.aging_tokens_per_ms * feature.waiting_time_ms
        - config.cache_reuse_weight * feature.cached_tokens,
    )
    if mode is SchedulingMode.CARAS:
        return (1, effective_work, *tie)

    if mode is SchedulingMode.SLO:
        slack = feature.slack_ms
        if slack is None:
            return (1, 1, effective_work, *tie)
        if slack >= 0:
            return (1, 0, slack, effective_work, *tie)
        # Among impossible requests, shortest remaining work is a useful
        # goodput-oriented fallback. Visible aged candidates get an override.
        return (1, 2, effective_work, -slack, *tie)

    raise AssertionError(f"unhandled scheduling mode: {mode}")
