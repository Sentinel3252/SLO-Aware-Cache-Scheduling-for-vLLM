"""Capture and compare deterministic outputs from baseline and CARAS servers."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any

from benchmark.schema import read_trace


def _complete(
    base_url: str, model: str, prompt: str, max_tokens: int
) -> dict[str, Any]:
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        payload = json.load(response)
    choice = payload["choices"][0]
    return {
        "text": choice["text"],
        "finish_reason": choice.get("finish_reason"),
    }


def capture(args: argparse.Namespace) -> None:
    if args.limit < 1:
        raise SystemExit("--limit must be >= 1")
    traces = read_trace(args.trace)[: args.limit]
    if not traces:
        raise SystemExit("cannot capture outputs from an empty trace")
    outputs = {
        trace.request_id: _complete(
            args.base_url,
            args.model,
            trace.prompt,
            trace.max_tokens,
        )
        for trace in traces
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(outputs, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"captured {len(outputs)} outputs to {args.output}")


def compare(args: argparse.Namespace) -> None:
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    for label, payload in (("baseline", baseline), ("candidate", candidate)):
        if not isinstance(payload, dict) or not payload:
            raise SystemExit(f"{label} must contain a nonempty request-id mapping")
    missing = sorted(set(baseline) ^ set(candidate))
    mismatches = sorted(
        request_id
        for request_id in set(baseline) & set(candidate)
        if baseline[request_id] != candidate[request_id]
    )
    if missing or mismatches:
        if missing:
            print("request-id set differs:", ", ".join(missing[:20]))
        if mismatches:
            print("output mismatches:", ", ".join(mismatches[:20]))
        raise SystemExit(1)
    print(f"output equivalence: OK ({len(baseline)} requests)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture_parser = subparsers.add_parser("capture")
    capture_parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    capture_parser.add_argument("--model", required=True)
    capture_parser.add_argument("--trace", type=Path, required=True)
    capture_parser.add_argument("--output", type=Path, required=True)
    capture_parser.add_argument("--limit", type=int, default=32)
    capture_parser.set_defaults(func=capture)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--baseline", type=Path, required=True)
    compare_parser.add_argument("--candidate", type=Path, required=True)
    compare_parser.set_defaults(func=compare)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
