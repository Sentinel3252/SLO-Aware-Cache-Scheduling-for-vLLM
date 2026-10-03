from __future__ import annotations

import argparse
import math
from pathlib import Path

from benchmark.schema import write_trace
from benchmark.workloads import cache_free, mixed, multiturn, rag


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a CacheFlow JSONL trace")
    parser.add_argument(
        "--workload", choices=("mixed", "rag", "multiturn", "cold"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--qps", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--slo-ms", type=float, default=1_000)
    parser.add_argument("--sessions", type=int, default=8)
    parser.add_argument("--turns", type=int, default=8)
    parser.add_argument(
        "--mixed-lengths",
        nargs="+",
        type=int,
        default=[128, 512, 2_048, 4_096],
        help="prompt lengths used by the mixed workload",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in ("qps", "slo_ms"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and > 0")
    if args.workload == "multiturn":
        if args.sessions < 1 or args.turns < 1:
            parser.error("--sessions and --turns must be >= 1")
    elif args.count < 1:
        parser.error("--count must be >= 1")
    if args.workload == "mixed" and any(length < 1 for length in args.mixed_lengths):
        parser.error("--mixed-lengths values must be >= 1")
    common = {"qps": args.qps, "seed": args.seed, "slo_ms": args.slo_ms}
    if args.workload == "mixed":
        requests = mixed(count=args.count, lengths=args.mixed_lengths, **common)
    elif args.workload == "rag":
        requests = rag(count=args.count, **common)
    elif args.workload == "cold":
        requests = cache_free(count=args.count, **common)
    else:
        requests = multiturn(
            sessions=args.sessions,
            turns=args.turns,
            qps=args.qps,
            seed=args.seed,
            slo_ms=args.slo_ms,
        )
    write_trace(args.output, requests)
    print(f"wrote {len(requests)} requests to {args.output}")


if __name__ == "__main__":
    main()
