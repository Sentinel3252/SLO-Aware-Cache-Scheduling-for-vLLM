# Getting started

## Install from source

Use Python 3.10+ for the core, simulator, and client. The vLLM runtime has its own
Python, accelerator, and platform requirements; install it in a compatible Linux
environment. Windows and macOS can run the CPU tools.

```bash
git clone https://github.com/Sentinel3252/SLO-Aware-Cache-Scheduling-for-vLLM.git
cd SLO-Aware-Cache-Scheduling-for-vLLM
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

In PowerShell, activate with `.venv\Scripts\Activate.ps1`. If activation is
unavailable, invoke `.venv\Scripts\python.exe` and commands under `.venv\Scripts\`
directly. The single-line CPU examples work in both shells.

```bash
python -m pip install -e ".[benchmark]"  # streaming client and plots
python -m pip install -e ".[dev]"        # lint and packaging tools
```

No PyPI release is assumed. A locally built wheel also works:
`python -m pip install dist/cacheflow_scheduler-0.1.0-py3-none-any.whl`.

## Compare policies on CPU

```bash
cacheflow-generate --workload rag --count 100 --qps 4 --seed 7 --output results/rag.jsonl
cacheflow-ablate --trace results/rag.jsonl --output-dir results/policies
```

The second command writes one JSONL file per policy plus `summary.md`. For a single mode:

```bash
cacheflow-simulate --trace results/rag.jsonl --mode residual --candidate-window 16 --output results/residual.jsonl
cacheflow-report --input Residual=results/residual.jsonl --output results/residual.md
```

`mixed`, `rag`, `multiturn`, and `cold` are supported generators. For `multiturn`,
count is `--sessions * --turns`; `--count` applies to the other generators.
Token estimates are synthetic, not tokenizer measurements.
For custom inputs, see the [JSONL trace fields and hand-written replay](trace-format.md).

`cacheflow-study --output-dir results/study` runs a larger seeded matrix.
Simulations omit GPU batching, finite KV capacity, eviction, decode contention,
and network latency. Label their reports as simulations.

## Run a vLLM server

Read [compatibility](compatibility.md), provision the GPU runtime, and install in
the same environment:

```bash
python -m pip install -r requirements/vllm.txt
cacheflow-check-vllm
export MODEL=Qwen/Qwen2.5-0.5B-Instruct
export CACHEFLOW_MODE=caras
export CACHEFLOW_CANDIDATE_WINDOW=16
export CACHEFLOW_METRICS_PATH=results/caras-scheduler.jsonl
vllm serve "$MODEL" \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --stream-interval 1 \
  --no-async-scheduling \
  --scheduler-cls cacheflow.vllm_adapter.CARASScheduler
```

From a checkout, `bash scripts/run_vllm.sh` supplies the same defaults and accepts
extra vLLM arguments. Restart the server to apply configuration changes.
Calibrate [cost estimates and targets](configuration.md) for the model/workload.

For an **upstream FCFS baseline or rollback**, stop the custom server and launch
with the same model and flags, replacing `--scheduler-cls ...` with
`--scheduling-policy fcfs`. CacheFlow environment variables do not affect the
upstream scheduler. `CACHEFLOW_MODE=fcfs` still uses the adapter and cache probes;
it is a policy comparison, not a zero-overhead upstream baseline.

## Measure a live server

In another terminal:

```bash
python -m pip install -e ".[benchmark]"
cacheflow-generate --workload rag --count 500 --qps 4 --seed 7 --slo-ms 1000 --output results/live-rag.jsonl
cacheflow-bench --base-url http://127.0.0.1:8000 --model Qwen/Qwen2.5-0.5B-Instruct --trace results/live-rag.jsonl --output results/live-caras.jsonl
```

Repeat against the FCFS server using the same trace, writing
`results/live-fcfs.jsonl`. Reset the server or use identical warmup for each run.
`--max-concurrency` must be at least one and `--timeout` must be finite and positive.

```bash
cacheflow-report --input FCFS=results/live-fcfs.jsonl --input CARAS=results/live-caras.jsonl --output results/live-summary.md
```

Check output equivalence separately with `cacheflow-outputs capture/compare`;
see [benchmark methodology](benchmark-methodology.md) for commands and metric limits.

## Command reference

| Command | Purpose | Additional runtime dependency |
| --- | --- | --- |
| `cacheflow-generate` | Generate JSONL traces | None |
| `cacheflow-simulate` | Simulate one policy | None |
| `cacheflow-ablate` | Compare all policies on a trace | None |
| `cacheflow-study` | Run a seeded simulation matrix | None |
| `cacheflow-bench` | Capture streaming metrics | `httpx` |
| `cacheflow-report` | Summarize metrics | `matplotlib` for `--plot` only |
| `cacheflow-outputs` | Capture/compare outputs | Reachable server for capture |
| `cacheflow-check-vllm` | Inspect installed interfaces | vLLM |

Every command accepts `--help` and works after wheel installation outside the
checkout. Original `python -m scripts.*` commands remain usable from a checkout.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| Command not found | Activate the installation environment; inspect its `bin` or `Scripts` directory. |
| vLLM not importable | Use the Linux GPU environment; inspect the original import error. |
| Unsupported internals | Confirm the exact vLLM version and follow the compatibility upgrade checks. |
| No scheduler JSONL yet | Emission occurs every N cycles; short runs may not reach the interval. |
| Poor tail latency | Compare upstream FCFS, inspect queue pressure and overhead, and tune window/age thresholds. |
