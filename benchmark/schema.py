from __future__ import annotations

import json
import math
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
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        request = cls(**value)
        if not isinstance(request.request_id, str) or not request.request_id.strip():
            raise ValueError("request_id must be a nonempty string")
        if not isinstance(request.prompt, str):
            raise ValueError("prompt must be a string")
        for field in ("prompt_tokens_estimate", "max_tokens"):
            count = getattr(request, field)
            if type(count) is not int or count <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if request.prefix_group is not None and not isinstance(
            request.prefix_group, str
        ):
            raise ValueError("prefix_group must be a string or null")
        _validate_time(request.arrival_ms, "arrival_ms", positive=False)
        if request.ttft_slo_ms is not None:
            _validate_time(request.ttft_slo_ms, "ttft_slo_ms", positive=True)
        return request


def _validate_time(value: object, field: str, *, positive: bool) -> None:
    if isinstance(value, float | int) and not isinstance(value, bool):
        try:
            if math.isfinite(value) and (value > 0 if positive else value >= 0):
                return
        except OverflowError:
            # Integers outside float range cannot be used by replay timing.
            pass
    allowed_range = "positive" if positive else "nonnegative"
    raise ValueError(f"{field} must be a finite {allowed_range} number")


def write_trace(path: Path, requests: list[TraceRequest]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for request in requests:
            handle.write(json.dumps(asdict(request), ensure_ascii=False) + "\n")


def read_trace(path: Path) -> list[TraceRequest]:
    requests: list[TraceRequest] = []
    first_lines: dict[str, int] = {}
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            try:
                line = raw_line.decode("utf-8")
                if not line.strip():
                    continue
                request = TraceRequest.from_dict(json.loads(line))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if request.request_id in first_lines:
                first = first_lines[request.request_id]
                raise ValueError(
                    f"{path}:{line_number}: duplicate request_id {request.request_id!r}"
                    f" (first defined on line {first})"
                )
            first_lines[request.request_id] = line_number
            requests.append(request)
    return requests
