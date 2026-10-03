from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class TraceRequest:
    request_id: str
    arrival_ms: float
    prompt: str
    prompt_tokens_estimate: int
    max_tokens: int = 16
    prefix_group: str | None = None
    ttft_slo_ms: float | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TraceRequest:
        return cls(**value)


def write_trace(path: Path, requests: list[TraceRequest]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for request in requests:
            handle.write(json.dumps(asdict(request), ensure_ascii=False) + "\n")


def read_trace(path: Path) -> list[TraceRequest]:
    with path.open("r", encoding="utf-8") as handle:
        return [
            TraceRequest.from_dict(json.loads(line)) for line in handle if line.strip()
        ]
