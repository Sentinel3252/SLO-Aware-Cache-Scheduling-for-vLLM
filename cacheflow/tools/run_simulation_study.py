"""Run a reproducible multi-workload simulation study.

The output is qualitative policy evidence, not a vLLM performance result.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections.abc import Callable, Sequence
from pathlib import Path

from benchmark.report import summarize
from benchmark.schema import TraceRequest
from benchmark.simulator import simulate
from benchmark.workloads import cache_free, mixed, multiturn, rag
from cacheflow.config import CARASConfig, SchedulingMode

METRICS = ("ttft_p95_ms", "ttft_p99_ms", "slo_attainment", "slo_goodput_rps")


def _mean_ci95(values: Sequence[float]) -> tuple[float, float]:
    """Return the sample mean and normal-approximation 95% half-width."""
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, math.nan
    return mean, 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def _workload(
    name: str, count: int, qps: float, seed: int, slo_ms: float
) -> list[TraceRequest]:
    factories: dict[str, Callable[..., list[TraceRequest]]] = {
        "mixed": mixed,
        "rag": rag,
        "cold": cache_free,
    }
    if name == "multiturn":
        sessions = min(8, count)
        turns = math.ceil(count / sessions)
        return multiturn(sessions, turns, qps, seed, slo_ms=slo_ms)[:count]
    return factories[name](count, qps, seed, slo_ms=slo_ms)


def run_study(
    workloads: Sequence[str],
    loads: Sequence[float],
    seeds: Sequence[int],
    modes: Sequence[SchedulingMode],
    count: int,
    slo_ms: float,
    candidate_windows: Sequence[int],
    prefill_tokens_per_ms: float,
    attention_context_weight: float,
) -> list[dict[str, float | int | str]]:
    """Run every requested cell and return one summary row per seed."""
    rows: list[dict[str, float | int | str]] = []
    for workload in workloads:
        for qps in loads:
            for seed in seeds:
                traces = _workload(workload, count, qps, seed, slo_ms)
                for candidate_window in candidate_windows:
                    for mode in modes:
                        summary = summarize(
                            simulate(
                                traces,
                                CARASConfig(
                                    mode=mode,
                                    candidate_window=candidate_window,
                                    prefill_tokens_per_ms=prefill_tokens_per_ms,
                                    attention_context_weight=attention_context_weight,
                                ),
                            )
                        )
                        rows.append(
                            {
                                "workload": workload,
                                "offered_qps": qps,
                                "seed": seed,
                                "candidate_window": candidate_window,
                                "policy": mode.value,
                                **summary,
                            }
                        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_report(rows: list[dict[str, float | int | str]]) -> str:
    """Aggregate seed repetitions and render a compact Markdown report."""
    cells: dict[tuple[str, float, int, str], list[dict[str, float | int | str]]] = {}
    for row in rows:
        key = (
            str(row["workload"]),
            float(row["offered_qps"]),
            int(row["candidate_window"]),
            str(row["policy"]),
        )
        cells.setdefault(key, []).append(row)

    lines = [
        "# Simulation study",
        "",
        "> These results come from the serial-prefill, infinite-cache ordering",
        "> simulator. They validate policy behavior only and are not vLLM/GPU",
        "> performance evidence.",
        "",
        "Values are mean ± normal-approximation 95% CI across seeds.",
        "",
        "| Workload | QPS | Window | Policy | TTFT p95 ms | TTFT p99 ms | "
        "SLO attainment | SLO goodput rps |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for key in sorted(cells):
        workload, qps, window, policy = key
        formatted = []
        for metric in METRICS:
            mean, ci = _mean_ci95([float(row[metric]) for row in cells[key]])
            formatted.append(f"{mean:.3f} ± {ci:.3f}")
        lines.append(
            f"| {workload} | {qps:g} | {window} | {policy} | "
            + " | ".join(formatted)
            + " |"
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--workload",
        action="append",
        choices=("mixed", "rag", "multiturn", "cold"),
        dest="workloads",
    )
    parser.add_argument("--load", action="append", type=float, dest="loads")
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--slo-ms", type=float, default=1_000)
    parser.add_argument("--window", action="append", type=int, dest="windows")
    parser.add_argument("--prefill-tokens-per-ms", type=float, default=4.0)
    parser.add_argument("--attention-context-weight", type=float, default=0.0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.count < 1:
        raise SystemExit("--count must be >= 1")
    workloads = args.workloads or ["mixed", "rag", "multiturn", "cold"]
    loads = args.loads or [1.0, 4.0, 16.0]
    seeds = args.seeds or [7, 17, 27]
    windows = args.windows or [1, 4, 16, 64]
    if any(load <= 0 for load in loads):
        raise SystemExit("--load values must be > 0")
    if any(window < 1 for window in windows):
        raise SystemExit("--window values must be >= 1")
    rows = run_study(
        workloads,
        loads,
        seeds,
        list(SchedulingMode),
        args.count,
        args.slo_ms,
        windows,
        args.prefill_tokens_per_ms,
        args.attention_context_weight,
    )
    _write_csv(args.output_dir / "runs.csv", rows)
    report = render_report(rows)
    (args.output_dir / "summary.md").write_text(report, encoding="utf-8")
    print(f"wrote {len(rows)} runs to {args.output_dir}")


if __name__ == "__main__":
    main()
