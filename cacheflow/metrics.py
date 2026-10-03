"""Low-overhead scheduler-policy metrics."""

from __future__ import annotations

import json
import threading
from bisect import bisect_left
from dataclasses import asdict, dataclass, field
from pathlib import Path

from cacheflow.models import ReorderResult

_REORDER_BUCKET_US = (
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1_024,
    4_096,
    1_000_000_000,
)


@dataclass(slots=True)
class MetricsSnapshot:
    cycles: int = 0
    candidates_scored: int = 0
    overtakes: int = 0
    reorder_ns: int = 0
    max_queue_length: int = 0
    reorder_histogram: list[int] = field(
        default_factory=lambda: [0] * len(_REORDER_BUCKET_US)
    )

    @property
    def mean_reorder_us(self) -> float:
        return self.reorder_ns / self.cycles / 1_000 if self.cycles else 0.0

    @property
    def p95_reorder_us_upper_bound(self) -> float:
        if not self.cycles:
            return 0.0
        target = self.cycles * 0.95
        cumulative = 0
        for upper_bound, count in zip(
            _REORDER_BUCKET_US, self.reorder_histogram, strict=True
        ):
            cumulative += count
            if cumulative >= target:
                return float(upper_bound)
        return float(_REORDER_BUCKET_US[-1])


class PolicyMetrics:
    def __init__(self, path: Path | None = None, interval: int = 100) -> None:
        self._snapshot = MetricsSnapshot()
        self._path = path
        self._interval = interval
        self._lock = threading.Lock()

    def record(self, result: ReorderResult) -> None:
        payload: dict[str, int | float] | None = None
        with self._lock:
            current = self._snapshot
            current.cycles += 1
            current.candidates_scored += result.candidates_scored
            current.overtakes += result.overtakes
            current.reorder_ns += result.elapsed_ns
            bucket = min(
                bisect_left(_REORDER_BUCKET_US, result.elapsed_ns / 1_000),
                len(_REORDER_BUCKET_US) - 1,
            )
            current.reorder_histogram[bucket] += 1
            current.max_queue_length = max(
                current.max_queue_length, result.queue_length
            )
            if self._path and current.cycles % self._interval == 0:
                payload = asdict(current)
                payload["mean_reorder_us"] = current.mean_reorder_us
                payload["p95_reorder_us_upper_bound"] = (
                    current.p95_reorder_us_upper_bound
                )
        if payload is not None:
            self._append(payload)

    def snapshot(self) -> MetricsSnapshot:
        with self._lock:
            s = self._snapshot
            return MetricsSnapshot(
                cycles=s.cycles,
                candidates_scored=s.candidates_scored,
                overtakes=s.overtakes,
                reorder_ns=s.reorder_ns,
                max_queue_length=s.max_queue_length,
                reorder_histogram=s.reorder_histogram.copy(),
            )

    def _append(self, payload: dict[str, int | float]) -> None:
        assert self._path is not None
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
