"""OpenAI-compatible streaming benchmark client with TTFT measurement."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from pathlib import Path
from typing import Any

from benchmark.metrics_collector import RequestMetric, write_metrics
from benchmark.schema import TraceRequest, read_trace


async def _issue_request(
    client: Any,
    base_url: str,
    model: str,
    request: TraceRequest,
    benchmark_start: float,
) -> RequestMetric:
    started = time.perf_counter()
    first_token_at: float | None = None
    token_times: list[float] = []
    output_tokens = 0
    status_code: int | None = None
    try:
        payload = {
            "model": model,
            "prompt": request.prompt,
            "max_tokens": request.max_tokens,
            "temperature": 0,
            "stream": True,
        }
        async with client.stream(
            "POST", f"{base_url.rstrip('/')}/v1/completions", json=payload
        ) as response:
            status_code = response.status_code
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                event = json.loads(line[6:])
                text = event.get("choices", [{}])[0].get("text", "")
                if text:
                    observed = time.perf_counter()
                    if first_token_at is None:
                        first_token_at = observed
                    token_times.append(observed)
                    output_tokens += 1
        ended = time.perf_counter()
        ttft = (first_token_at - started) * 1_000 if first_token_at else None
        tpot = None
        if len(token_times) > 1:
            tpot = (token_times[-1] - token_times[0]) * 1_000 / (len(token_times) - 1)
        return RequestMetric(
            request_id=request.request_id,
            ok=True,
            status_code=status_code,
            scheduled_at_ms=(started - benchmark_start) * 1_000,
            ttft_ms=ttft,
            tpot_ms=tpot,
            e2e_ms=(ended - started) * 1_000,
            output_tokens=output_tokens,
            slo_ms=request.ttft_slo_ms,
        )
    except Exception as exc:  # retain failed requests in goodput denominator
        ended = time.perf_counter()
        return RequestMetric(
            request_id=request.request_id,
            ok=False,
            status_code=status_code,
            scheduled_at_ms=(started - benchmark_start) * 1_000,
            ttft_ms=None,
            tpot_ms=None,
            e2e_ms=(ended - started) * 1_000,
            output_tokens=0,
            slo_ms=request.ttft_slo_ms,
            error=f"{type(exc).__name__}: {exc}",
        )


async def run(args: argparse.Namespace) -> list[RequestMetric]:
    if args.max_concurrency < 1:
        raise SystemExit("--max-concurrency must be >= 1")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise SystemExit("--timeout must be finite and > 0")
    try:
        import httpx
    except ImportError as exc:
        raise SystemExit(
            "install benchmark dependencies: pip install -e '.[benchmark]'"
        ) from exc
    requests = read_trace(args.trace)
    limits = httpx.Limits(
        max_connections=args.max_concurrency,
        max_keepalive_connections=args.max_concurrency,
    )
    semaphore = asyncio.Semaphore(args.max_concurrency)
    async with httpx.AsyncClient(timeout=args.timeout, limits=limits) as client:
        started = time.perf_counter() + 0.25

        async def bounded(request: TraceRequest) -> RequestMetric:
            target = started + request.arrival_ms / 1_000
            await asyncio.sleep(max(0.0, target - time.perf_counter()))
            async with semaphore:
                return await _issue_request(
                    client, args.base_url, args.model, request, started
                )

        return await asyncio.gather(*(bounded(request) for request in requests))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark a vLLM OpenAI server")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--max-concurrency", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=600)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    metrics = asyncio.run(run(args))
    write_metrics(args.output, metrics)
    failures = sum(not metric.ok for metric in metrics)
    print(f"wrote {len(metrics)} results ({failures} failures) to {args.output}")


if __name__ == "__main__":
    main()
