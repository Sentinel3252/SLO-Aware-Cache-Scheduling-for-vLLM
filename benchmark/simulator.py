"""Admission-order simulator for fast policy sanity checks.

This is intentionally not a vLLM performance model.  It isolates waiting-queue
ordering under a serial prefill server and an infinite prefix cache, making it
useful for deterministic tests and qualitative ablations only.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmark.schema import TraceRequest, read_trace
from cacheflow.config import CARASConfig, SchedulingMode
from cacheflow.models import RequestFeature
from cacheflow.request_queue import CARASRequestQueue


@dataclass(eq=False, slots=True)
class SimRequest:
    trace: TraceRequest
    request_id: str
    arrival_time: float
    priority: int = 0
    num_preemptions: int = 0


class SimFeatureProvider:
    def __init__(
        self,
        config: CARASConfig,
        cache: dict[str, int],
        block_size: int = 16,
    ) -> None:
        self.config = config
        self.cache = cache
        self.block_size = block_size

    def __call__(self, request: SimRequest, now: float | None) -> RequestFeature:
        assert now is not None
        trace = request.trace
        cached = 0
        if trace.prefix_group:
            cached = min(
                self.cache.get(trace.prefix_group, 0),
                max(0, trace.prompt_tokens_estimate - 1),
            )
            if cached < self.config.min_cache_blocks * self.block_size:
                cached = 0
        remaining = max(1, trace.prompt_tokens_estimate - cached)
        waiting_ms = max(0.0, now * 1_000 - trace.arrival_ms)
        context = cached + remaining / 2
        equivalent_tokens = remaining * (
            1
            + self.config.attention_context_weight
            * context
            / self.config.attention_reference_tokens
        )
        return RequestFeature(
            request_id=request.request_id,
            arrival_time=request.arrival_time,
            priority=request.priority,
            prompt_tokens=trace.prompt_tokens_estimate,
            cached_tokens=cached,
            remaining_prefill_tokens=remaining,
            waiting_time_ms=waiting_ms,
            estimated_prefill_ms=(
                equivalent_tokens / self.config.prefill_tokens_per_ms
            ),
            required_kv_blocks=math.ceil(remaining / self.block_size),
            slo_ms=trace.ttft_slo_ms,
        )


def simulate(traces: list[TraceRequest], config: CARASConfig) -> list[dict[str, Any]]:
    cache: dict[str, int] = {}
    provider = SimFeatureProvider(config, cache)
    queue = CARASRequestQueue("fcfs", config, provider)
    arrivals = [
        (
            trace.arrival_ms / 1_000,
            index,
            SimRequest(trace, trace.request_id, trace.arrival_ms / 1_000),
        )
        for index, trace in enumerate(traces)
    ]
    heapq.heapify(arrivals)
    now = 0.0
    rows: list[dict[str, Any]] = []
    while arrivals or queue:
        if not queue:
            now = max(now, arrivals[0][0])
        while arrivals and arrivals[0][0] <= now:
            _arrival, _index, request = heapq.heappop(arrivals)
            queue.add_request(request)
        queue.prepare(now)
        request = queue.pop_request()
        feature = provider(request, now)
        started = now
        service_ms = feature.estimated_prefill_ms
        now += service_ms / 1_000
        ttft_ms = (started - request.arrival_time) * 1_000 + service_ms
        trace = request.trace
        if trace.prefix_group:
            cache[trace.prefix_group] = max(
                cache.get(trace.prefix_group, 0), trace.prompt_tokens_estimate
            )
        rows.append(
            {
                "request_id": request.request_id,
                "ok": True,
                "status_code": 200,
                # Match the online client's schema: scheduled_at is the issue
                # time and e2e includes queueing through first token.
                "scheduled_at_ms": request.trace.arrival_ms,
                "ttft_ms": ttft_ms,
                "tpot_ms": 0.0,
                "e2e_ms": ttft_ms,
                "output_tokens": trace.max_tokens,
                "slo_ms": trace.ttft_slo_ms,
                "slo_met": trace.ttft_slo_ms is not None
                and ttft_ms <= trace.ttft_slo_ms,
                "cached_tokens": feature.cached_tokens,
                "remaining_prefill_tokens": feature.remaining_prefill_tokens,
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the CacheFlow ordering simulator")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=[mode.value for mode in SchedulingMode], default="caras"
    )
    parser.add_argument("--candidate-window", type=int, default=16)
    parser.add_argument("--max-wait-ms", type=float, default=2_000)
    parser.add_argument("--prefill-tokens-per-ms", type=float, default=4.0)
    parser.add_argument("--aging-tokens-per-ms", type=float, default=2.0)
    parser.add_argument("--attention-context-weight", type=float, default=0.0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = CARASConfig(
        mode=SchedulingMode(args.mode),
        candidate_window=args.candidate_window,
        max_wait_ms=args.max_wait_ms,
        prefill_tokens_per_ms=args.prefill_tokens_per_ms,
        aging_tokens_per_ms=args.aging_tokens_per_ms,
        attention_context_weight=args.attention_context_weight,
    )
    rows = simulate(read_trace(args.trace), config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"wrote {len(rows)} simulated requests to {args.output}")


if __name__ == "__main__":
    main()
