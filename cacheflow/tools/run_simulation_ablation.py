from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.report import markdown_table, summarize
from benchmark.schema import read_trace
from benchmark.simulator import simulate
from cacheflow.config import CARASConfig, SchedulingMode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all policy modes in the analytical simulator"
    )
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-window", type=int, default=16)
    parser.add_argument("--prefill-tokens-per-ms", type=float, default=4.0)
    parser.add_argument("--attention-context-weight", type=float, default=0.0)
    args = parser.parse_args()

    traces = read_trace(args.trace)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for mode in SchedulingMode:
        rows = simulate(
            traces,
            CARASConfig(
                mode=mode,
                candidate_window=args.candidate_window,
                prefill_tokens_per_ms=args.prefill_tokens_per_ms,
                attention_context_weight=args.attention_context_weight,
            ),
        )
        path = args.output_dir / f"{mode.value}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        summaries.append((mode.value, summarize(rows)))
    table = (
        "# Policy comparison (simulation only)\n\n"
        "Serial prefill and unlimited prefix cache; not GPU performance evidence.\n\n"
        + markdown_table(summaries)
    )
    (args.output_dir / "summary.md").write_text(table, encoding="utf-8")
    print(table, end="")


if __name__ == "__main__":
    main()
