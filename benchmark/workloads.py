"""Deterministic synthetic workload construction."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from benchmark.schema import TraceRequest

_WORDS = (
    "amber bridge cedar delta ember forest galaxy harbor ivory juniper "
    "kernel lantern meadow nebula orchard prism quartz river summit timber"
).split()


def tokenish_text(tokens: int, salt: int = 0) -> str:
    """Create stable whitespace-separated text with roughly ``tokens`` BPE tokens."""

    return " ".join(_WORDS[(index + salt) % len(_WORDS)] for index in range(tokens))


def poisson_arrivals(count: int, qps: float, rng: random.Random) -> list[float]:
    if not math.isfinite(qps) or qps <= 0:
        raise ValueError("qps must be finite and > 0")
    elapsed = 0.0
    arrivals: list[float] = []
    for _ in range(count):
        elapsed += rng.expovariate(qps) * 1_000
        arrivals.append(elapsed)
    return arrivals


def mixed(
    count: int,
    qps: float,
    seed: int,
    lengths: Sequence[int] = (128, 512, 2_048, 8_192),
    slo_ms: float | None = 1_000,
) -> list[TraceRequest]:
    rng = random.Random(seed)
    arrivals = poisson_arrivals(count, qps, rng)
    return [
        TraceRequest(
            request_id=f"mixed-{index:05d}",
            arrival_ms=arrival,
            prompt=tokenish_text(length, salt=index),
            prompt_tokens_estimate=length,
            ttft_slo_ms=slo_ms,
        )
        for index, (arrival, length) in enumerate(
            zip(arrivals, (rng.choice(lengths) for _ in range(count)), strict=True)
        )
    ]


def rag(
    count: int,
    qps: float,
    seed: int,
    prefix_tokens: int = 2_048,
    suffix_tokens: int = 128,
    groups: int = 8,
    slo_ms: float | None = 1_000,
) -> list[TraceRequest]:
    if groups < 1:
        raise ValueError("groups must be >= 1")
    if prefix_tokens < 1:
        raise ValueError("prefix_tokens must be >= 1")
    rng = random.Random(seed)
    arrivals = poisson_arrivals(count, qps, rng)
    # A group marker replaces one synthetic word. Rotating the short vocabulary
    # alone repeats every 20 groups and can make unrelated documents identical.
    prefixes = [
        f"doc-{group} " + tokenish_text(prefix_tokens - 1, salt=group * 3)
        for group in range(groups)
    ]
    requests = []
    for index, arrival in enumerate(arrivals):
        group = rng.randrange(groups)
        suffix = tokenish_text(suffix_tokens, salt=10_000 + index)
        requests.append(
            TraceRequest(
                request_id=f"rag-{index:05d}",
                arrival_ms=arrival,
                prompt=f"{prefixes[group]}\nQuestion {index}: {suffix}",
                prompt_tokens_estimate=prefix_tokens + suffix_tokens + 3,
                prefix_group=f"doc-{group}",
                ttft_slo_ms=slo_ms,
            )
        )
    return requests


def multiturn(
    sessions: int,
    turns: int,
    qps: float,
    seed: int,
    system_tokens: int = 512,
    turn_tokens: int = 64,
    slo_ms: float | None = 1_000,
) -> list[TraceRequest]:
    rng = random.Random(seed)
    count = sessions * turns
    arrivals = poisson_arrivals(count, qps, rng)
    histories = [
        tokenish_text(system_tokens, salt=index * 7) for index in range(sessions)
    ]
    requests: list[TraceRequest] = []
    order = [(turn, session) for turn in range(turns) for session in range(sessions)]
    for (turn, session), arrival in zip(order, arrivals, strict=True):
        histories[session] += " " + tokenish_text(
            turn_tokens, salt=50_000 + turn * sessions + session
        )
        requests.append(
            TraceRequest(
                request_id=f"session-{session:03d}-turn-{turn:03d}",
                arrival_ms=arrival,
                prompt=histories[session],
                prompt_tokens_estimate=system_tokens + (turn + 1) * turn_tokens,
                prefix_group=f"session-{session}",
                ttft_slo_ms=slo_ms,
            )
        )
        # A model response is represented in the next turn's shared history.
        histories[session] += " assistant acknowledgement"
    return requests


def cache_free(
    count: int,
    qps: float,
    seed: int,
    prompt_tokens: int = 1_024,
    slo_ms: float | None = 1_000,
) -> list[TraceRequest]:
    rng = random.Random(seed)
    arrivals = poisson_arrivals(count, qps, rng)
    requests = []
    for index, arrival in enumerate(arrivals):
        # The UUID-like first line forces a different first cache block.
        nonce = f"unique-{seed:08x}-{index:016x}"
        body = tokenish_text(prompt_tokens, salt=index * 997)
        requests.append(
            TraceRequest(
                request_id=f"cold-{index:05d}",
                arrival_ms=arrival,
                prompt=f"{nonce} {body}",
                prompt_tokens_estimate=prompt_tokens + 1,
                ttft_slo_ms=slo_ms,
            )
        )
    return requests


def offered_duration_s(requests: Sequence[TraceRequest]) -> float:
    if not requests:
        return 0.0
    return math.ceil(max(request.arrival_ms for request in requests)) / 1_000
