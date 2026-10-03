from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RequestMetric:
    request_id: str
    ok: bool
    status_code: int | None
    scheduled_at_ms: float
    ttft_ms: float | None
    tpot_ms: float | None
    e2e_ms: float
    output_tokens: int
    slo_ms: float | None
    error: str | None = None

    @property
    def slo_met(self) -> bool | None:
        if self.slo_ms is None or self.ttft_ms is None:
            return None
        return self.ok and self.ttft_ms <= self.slo_ms


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    rank = (len(ordered) - 1) * quantile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def write_metrics(path: Path, metrics: list[RequestMetric]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for metric in metrics:
            payload = asdict(metric)
            payload["slo_met"] = metric.slo_met
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
