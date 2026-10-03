"""Validate built archives and exercise installed CLIs outside the checkout."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tarfile
import tempfile
import venv
import zipfile
from pathlib import Path

COMMANDS = (
    "cacheflow-generate",
    "cacheflow-simulate",
    "cacheflow-report",
    "cacheflow-bench",
    "cacheflow-check-vllm",
    "cacheflow-outputs",
    "cacheflow-ablate",
    "cacheflow-study",
)


def check_archives(wheel: Path, source: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        wheel_names = archive.namelist()
        required = {
            "cacheflow/tools/check_vllm_compat.py",
            "cacheflow/tools/output_equivalence.py",
            "cacheflow/tools/run_simulation_ablation.py",
            "cacheflow/tools/run_simulation_study.py",
            "benchmark/workload_generator.py",
        }
        missing = required - set(wheel_names)
        if missing:
            raise RuntimeError(f"wheel missing modules: {sorted(missing)}")
    with tarfile.open(source) as archive:
        source_names = [name.partition("/")[2] for name in archive.getnames()]
    required_docs = {
        "LICENSE",
        "README.md",
        "README.zh-CN.md",
        "CONTRIBUTING.md",
        "docs/getting-started.md",
        "requirements/vllm.txt",
        "scripts/check_distribution.py",
    }
    missing_docs = required_docs - set(source_names)
    if missing_docs:
        raise RuntimeError(f"source archive missing files: {sorted(missing_docs)}")
    for name in [*wheel_names, *source_names]:
        parts = Path(name).parts
        if any(part in {"interview", ".local", "__pycache__"} for part in parts):
            raise RuntimeError(f"personal or generated file in archive: {name}")
        if name.startswith(("results/", "benchmark/traces/")):
            raise RuntimeError(f"generated trace/result in archive: {name}")


def run_smoke(wheel: Path) -> None:
    # Do not inherit source-tree imports into the installation environment.
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    with tempfile.TemporaryDirectory(prefix="cacheflow-wheel-") as raw_temp:
        root = Path(raw_temp)
        env_path = root / "env"
        venv.EnvBuilder(with_pip=True).create(env_path)
        binary_dir = env_path / ("Scripts" if os.name == "nt" else "bin")
        python = binary_dir / ("python.exe" if os.name == "nt" else "python")

        def run(*args: str | Path) -> None:
            subprocess.run(
                [str(arg) for arg in args],
                check=True,
                cwd=root,
                env=environment,
                timeout=120,
            )

        def cli(command: str, *args: str) -> None:
            suffix = ".exe" if os.name == "nt" else ""
            run(binary_dir / f"{command}{suffix}", *args)

        run(python, "-m", "pip", "install", "--no-index", "--no-deps", wheel)
        for command in COMMANDS:
            cli(command, "--help")
        cli(
            "cacheflow-generate",
            "--workload",
            "rag",
            "--count",
            "12",
            "--qps",
            "4",
            "--seed",
            "7",
            "--output",
            "trace.jsonl",
        )
        for mode in ("fcfs", "caras"):
            cli(
                "cacheflow-simulate",
                "--trace",
                "trace.jsonl",
                "--mode",
                mode,
                "--output",
                f"{mode}.jsonl",
            )
            rows = [
                json.loads(line)
                for line in (root / f"{mode}.jsonl").read_text().splitlines()
            ]
            if len(rows) != 12 or not all(row["ok"] for row in rows):
                raise RuntimeError(f"invalid {mode} simulation output")
        cli(
            "cacheflow-report",
            "--input",
            "FCFS=fcfs.jsonl",
            "--input",
            "CARAS=caras.jsonl",
            "--output",
            "summary.md",
        )
        if "CARAS" not in (root / "summary.md").read_text():
            raise RuntimeError("report did not contain candidate policy")
        cli("cacheflow-ablate", "--trace", "trace.jsonl", "--output-dir", "policies")
        cli(
            "cacheflow-study",
            "--workload",
            "cold",
            "--count",
            "8",
            "--load",
            "1",
            "--seed",
            "7",
            "--window",
            "1",
            "--output-dir",
            "study",
        )
        if not (root / "study" / "runs.csv").is_file():
            raise RuntimeError("study did not write runs.csv")
        for mode in ("fcfs", "cache", "residual", "aging", "caras", "slo"):
            if not (root / "policies" / f"{mode}.jsonl").is_file():
                raise RuntimeError(f"policy comparison missing {mode}")
        (root / "baseline.json").write_text('{"r1": {"text": "ok"}}')
        (root / "candidate.json").write_text('{"r1": {"text": "ok"}}')
        cli(
            "cacheflow-outputs",
            "compare",
            "--baseline",
            "baseline.json",
            "--candidate",
            "candidate.json",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    args = parser.parse_args()
    wheels = sorted(args.dist_dir.glob("cacheflow_scheduler-*.whl"))
    sources = sorted(args.dist_dir.glob("cacheflow_scheduler-*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        raise SystemExit("use a fresh dist directory with one wheel and one sdist")
    check_archives(wheels[0], sources[0])
    run_smoke(wheels[0].resolve())
    print("distribution validation: OK")


if __name__ == "__main__":
    main()
