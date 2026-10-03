# Benchmark methodology

The benchmark should show whether changing admission order improves latency,
which part of the policy causes the change, and whether it adds CPU overhead or
makes other requests wait longer.

## Comparison setup

- Pin the exact vLLM build, model revision, tokenizer, hardware, and GPU runtime.
- Use upstream FCFS as the baseline; CacheFlow `mode=fcfs` includes adapter overhead.
- Use `--no-async-scheduling` and `--stream-interval 1` for every compared server.
- Replay the same trace, seeds, and arrivals; reset the server or use identical warmup.
- Fix model length, batch-token budget, sequence limit, block size, and GPU memory.
- Sweep offered load around saturation and collect at least three seeds per point.
- Record failed requests, preemption/OOM, competing GPU activity, and throttling.

## Inputs

Include mixed prompt lengths, distinct RAG prefix groups, multi-session continuation,
and unique-prefix cold workloads. Testing a single resident prefix tells us little
about cache residency contention. Validate actual token counts with the target tokenizer;
`prompt_tokens_estimate` is only synthetic generator metadata.
The [trace format guide](trace-format.md) documents fields and custom replay.
RAG document markers keep groups textually distinct beyond the synthetic word
vocabulary; validate actual tokenized prefix separation on the target model.

Trace-level targets are used for reporting and simulation; online scheduler targets
come from `CACHEFLOW_TTFT_SLO_MS`. Set them consistently when evaluating SLO mode.

## Correctness checks

```bash
python -m unittest discover -v
cacheflow-check-vllm
export MODEL=Qwen/Qwen2.5-0.5B-Instruct
cacheflow-outputs capture --base-url http://127.0.0.1:8000 --model "$MODEL" --trace results/rag.jsonl --output results/fcfs-outputs.json
```

Capture against upstream FCFS, restart as CARAS with identical model settings, then:

```bash
cacheflow-outputs capture --base-url http://127.0.0.1:8000 --model "$MODEL" --trace results/rag.jsonl --output results/caras-outputs.json
cacheflow-outputs compare --baseline results/fcfs-outputs.json --candidate results/caras-outputs.json
```

Capture uses temperature zero and defaults to the first 32 trace requests; increase
`--limit` to cover the desired trace. This checks completion text and finish reason
through sequential requests. Capture rejects empty traces and nonpositive limits;
comparison requires nonempty request-ID mappings on both sides. It does not validate
concurrent cancellation, preemption, or sampled token-ID equivalence. Exercise those
cases separately.

## Metrics and current measurement limits

The report computes successful-request TTFT p50/p95/p99, TPOT p95, E2E p95, request
throughput, SLO attainment, and goodput. Failures remain in the SLO denominator when
a target is present. Goodput is successful targeted requests meeting TTFT divided
by run duration; no-target rows do not count as SLO successes.

The streaming client starts timing when it issues the HTTP request. Waiting for
its concurrency semaphore is excluded from TTFT, so ensure concurrency does not
throttle intended arrivals. Scheduled timestamps are actual issue times, not trace
arrival times; record client-side admission lag separately when assessing load.

`output_tokens` currently counts nonempty SSE text chunks. TPOT is the mean interval
between those chunks. `--stream-interval 1` helps, but chunks are not guaranteed to
be individual tokens. Treat TPOT as an approximation and do not claim exact output
token throughput from this field; retain server/tokenizer counts if needed.

Scheduler snapshots provide candidate counts, overtakes, mean preparation time,
a histogram p95 upper bound, and maximum queue size. They do not provide maximum
request wait, cache hit ratios, preemption counts, or KV utilization. Gather these
from engine instrumentation and report metric-write overhead separately.

## Report and publish the results

Report each load point independently with sample counts, seeds, mean, standard
deviation, and confidence intervals. Bootstrap tail samples when appropriate.
Do not select only the best run or combine unrelated loads into one percentile.

Publish sanitized traces or generator commands, raw metrics, environment/model
revisions, flags, configuration, and checksums alongside summaries. If the raw
files are missing, say so; readers need them to reproduce the comparison. Report
regressions too, and identify any fairness or overhead measurements you did not collect.

`cacheflow-study --output-dir results/study` produces a qualitative CPU matrix
with normal-approximation confidence intervals across seeds. Its serial prefill
server and unlimited cache do not model vLLM GPU performance.
