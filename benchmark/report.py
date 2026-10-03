from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from benchmark.metrics_collector import percentile


def load_metrics(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    good = [row for row in rows if row.get("ok")]
    ttft = [float(row["ttft_ms"]) for row in good if row.get("ttft_ms") is not None]
    tpot = [float(row["tpot_ms"]) for row in good if row.get("tpot_ms") is not None]
    e2e = [float(row["e2e_ms"]) for row in good]
    end_times = [float(row["scheduled_at_ms"]) + float(row["e2e_ms"]) for row in rows]
    duration_s = max(end_times, default=0.0) / 1_000
    slo_rows = [row for row in rows if row.get("slo_ms") is not None]
    slo_met = sum(bool(row.get("slo_met")) for row in slo_rows)

    def p(values: list[float], q: float) -> float:
        return percentile(values, q) if values else math.nan

    return {
        "requests": len(rows),
        "failures": len(rows) - len(good),
        "ttft_p50_ms": p(ttft, 0.50),
        "ttft_p95_ms": p(ttft, 0.95),
        "ttft_p99_ms": p(ttft, 0.99),
        "tpot_p95_ms": p(tpot, 0.95),
        "e2e_p95_ms": p(e2e, 0.95),
        "slo_attainment": slo_met / len(slo_rows) if slo_rows else math.nan,
        "slo_goodput_rps": slo_met / duration_s if duration_s else 0.0,
        "request_throughput_rps": len(good) / duration_s if duration_s else 0.0,
    }


def markdown_table(summaries: list[tuple[str, dict[str, float | int]]]) -> str:
    columns = [
        "policy",
        "requests",
        "failures",
        "ttft p50 ms",
        "ttft p95 ms",
        "ttft p99 ms",
        "SLO attainment",
        "SLO goodput rps",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for label, summary in summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    str(summary["requests"]),
                    str(summary["failures"]),
                    f"{summary['ttft_p50_ms']:.2f}",
                    f"{summary['ttft_p95_ms']:.2f}",
                    f"{summary['ttft_p99_ms']:.2f}",
                    f"{summary['slo_attainment']:.3f}",
                    f"{summary['slo_goodput_rps']:.3f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def plot(summaries: list[tuple[str, dict[str, float | int]]], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required for --plot") from exc
    labels = [label for label, _summary in summaries]
    values = [float(summary["slo_goodput_rps"]) for _label, summary in summaries]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(labels, values, color="#3b82f6")
    axis.set_ylabel("TTFT SLO goodput (requests/s)")
    axis.set_title("CacheFlow policy comparison")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize CacheFlow benchmark results"
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="repeat for each policy",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    summaries = []
    for value in args.input:
        label, separator, raw_path = value.partition("=")
        if not separator:
            raise SystemExit(f"invalid --input {value!r}; expected LABEL=PATH")
        summaries.append((label, summarize(load_metrics(Path(raw_path)))))
    table = markdown_table(summaries)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(table, encoding="utf-8")
    else:
        print(table, end="")
    if args.plot:
        plot(summaries, args.plot)


if __name__ == "__main__":
    main()
