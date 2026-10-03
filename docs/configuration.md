# Configuration

The adapter reads `CACHEFLOW_*` environment variables at startup. Restart the server
after changes. Pure Python callers can construct `CARASConfig` directly.

## Policies

Lower keys are admitted first. Priority is an outer key in priority mode, and
ties use `(arrival_time, request_id)`.

| Mode | Ranking |
| --- | --- |
| `fcfs` | Arrival time, request ID |
| `cache` | Descending cached tokens |
| `residual` | Ascending remaining prefill tokens |
| `aging` | `max(0, remaining - aging_rate * wait_ms)` |
| `caras` | Estimated work minus age and optional cache credit |
| `slo` | Feasible targets by slack, then untargeted requests, then infeasible targets by effective work |

All modes apply the max-wait override to visible candidates first. It cannot
override priority classes, inspect the queue tail, or force resource allocation.
It is not a hard waiting-time bound.

## Environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `CACHEFLOW_MODE` | `caras` | One of the modes above |
| `CACHEFLOW_CANDIDATE_WINDOW` | `16` | Candidates scored per cycle; integer >= 1 |
| `CACHEFLOW_MAX_WAIT_MS` | `2000` | Age override threshold; > 0 |
| `CACHEFLOW_AGING_TOKENS_PER_MS` | `2.0` | Age credit rate; >= 0 |
| `CACHEFLOW_CACHE_REUSE_WEIGHT` | `0.0` | Extra credit per cached token; >= 0 |
| `CACHEFLOW_MIN_CACHE_BLOCKS` | `1` | Treat smaller hits as misses; integer >= 0 |
| `CACHEFLOW_PREFILL_TOKENS_PER_MS` | `4.0` | Calibrated service rate; > 0 |
| `CACHEFLOW_ATTENTION_CONTEXT_WEIGHT` | `0.0` | Attention-context correction; >= 0 |
| `CACHEFLOW_ATTENTION_REFERENCE_TOKENS` | `4096` | Context normalization length; integer >= 1 |
| `CACHEFLOW_TTFT_SLO_MS` | unset | Global TTFT target in ms; > 0 when set |
| `CACHEFLOW_PROTECT_PREEMPTED` | `true` | FCFS preempted front protection |
| `CACHEFLOW_METRICS_PATH` | unset | Scheduler aggregate JSONL path |
| `CACHEFLOW_METRICS_INTERVAL` | `100` | Emit every N cycles; integer >= 1 |

Booleans accept `1/true/yes/on` and `0/false/no/off`, case-insensitively.
Invalid modes, numeric conversions, and out-of-range parameters fail at startup.
Floating-point policy parameters must also be finite: NaN and positive or negative
infinity are rejected, both for environment variables and direct `CARASConfig`
construction.

## Calibration

Residual work already subtracts cached tokens. Keep extra cache credit at zero
unless comparing additional cache affinity. The context correction approximates
suffix attention over a cached prefix; it is not a measured service-time model.
Measure prefill timing on the target model/GPU before interpreting SLO rankings.
The default rate is a starting parameter, not a hardware throughput claim.

Set `CACHEFLOW_TTFT_SLO_MS` explicitly for online SLO mode. The OpenAI-compatible
client does not transport a private per-request CacheFlow target; trace-level
targets affect client reporting and simulation only.

Start with a small window and compare upstream FCFS. Evaluate tails, queue
pressure, and fairness together. Increasing K adds choices and scoring work;
K=1 prevents reordering but still incurs adapter work.

## Telemetry

Set `CACHEFLOW_METRICS_PATH` for cumulative JSONL snapshots of cycles, candidates,
overtakes, reorder nanoseconds, maximum queue length, a reorder histogram, mean
reorder microseconds, and a bucketed p95 upper bound. Do not sum successive rows.

The p95 is a histogram bound, not an exact percentile. Overtakes measure forward
position changes, not starvation. Maximum request wait, preemptions, and KV
utilization require separate engine instrumentation. Partial intervals are not
flushed on shutdown. Use a writable path per process; JSONL writes are synchronous
and must be included when measuring operational overhead.
