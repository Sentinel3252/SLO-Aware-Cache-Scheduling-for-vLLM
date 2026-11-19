# JSONL trace format

CacheFlow generators, the ordering simulator, streaming benchmark client, and
output capture tool share a trace format: one JSON object per nonblank line,
encoded as UTF-8. Use a unique request ID for every request in a file.

## Fields

| Field | Type | Required / default | Meaning |
| --- | --- | --- | --- |
| `request_id` | nonempty string | Required | Unique request identity used in queues and output comparison |
| `arrival_ms` | number | Required | Planned arrival in milliseconds relative to the start of replay |
| `prompt` | string | Required | Prompt sent to the completion endpoint |
| `prompt_tokens_estimate` | integer | Required | Synthetic or externally measured prompt length used by the simulator |
| `max_tokens` | integer | `16` | Completion length limit; simulation does not model decode work |
| `prefix_group` | string or null | `null` | Shared-prefix identity used by the simulator |
| `ttft_slo_ms` | number or null | `null` | TTFT target for simulation ranking and client reporting |

Use finite, nonnegative arrival times, positive integer token counts, and finite,
positive SLO targets when a target is set. JSON booleans are invalid as numbers.
JSON keys must match the fields above. Omitted optional fields use their defaults.
Blank lines are ignored, and arrival times do not have to be sorted in the file.

Replay validates the entire trace before starting. Invalid records report the
path and physical line number. When an ID appears twice, the error identifies both
its original line and the line that repeats it.

`prompt_tokens_estimate` is not sent as an HTTP token count. Validate prompt lengths
with the target tokenizer when measuring a real server. The synthetic word counts
in generated traces are approximate metadata, including document identity markers.

The simulator infers reuse from `prefix_group`, rather than comparing actual prompt
tokens. The online client sends the prompt itself; it does not send `prefix_group`
or a private per-request scheduler SLO. Online scheduler targets are configured
with `CACHEFLOW_TTFT_SLO_MS`. Keep shared groups consistent with shared text when
preparing traces intended for both simulation and live measurement.

## Hand-written replay

Save the following as `results/manual.jsonl`, creating the directory if needed:

```jsonl
{"request_id":"doc-a-1","arrival_ms":0,"prompt":"Shared document A. First question?","prompt_tokens_estimate":12,"max_tokens":8,"prefix_group":"doc-a","ttft_slo_ms":100}
{"request_id":"doc-a-2","arrival_ms":1,"prompt":"Shared document A. Second question?","prompt_tokens_estimate":14,"max_tokens":8,"prefix_group":"doc-a","ttft_slo_ms":100}
{"request_id":"cold-1","arrival_ms":2,"prompt":"An unrelated prompt.","prompt_tokens_estimate":6}
```

From an installed CacheFlow environment, run:

```bash
cacheflow-simulate --trace results/manual.jsonl --mode fcfs --output results/manual-fcfs.jsonl
cacheflow-simulate --trace results/manual.jsonl --mode residual --output results/manual-residual.jsonl
cacheflow-report --input FCFS=results/manual-fcfs.jsonl --input Residual=results/manual-residual.jsonl
```

Each simulation produces three request rows. These are serial-prefill,
infinite-cache ordering results, not GPU latency measurements. Output capture
replays requests sequentially and ignores arrival timing.

The `multiturn` generator creates synthetic session histories with assistant
acknowledgement placeholders. Live replay uses the prebuilt prompts and planned
arrivals; it does not wait for actual model replies before building the next turn.

## Generator validation and reproducibility

The generator CLI requires finite, positive QPS and SLO values. Mixed lengths
must be positive. Multiturn sessions and turns must each be at least one; its
request count is their product, and `--count` applies to the other workloads.

RAG prefixes include a stable document marker so groups remain distinct beyond
the twenty-word synthetic vocabulary. This changed generated prompt text: keep
the CacheFlow version or commit with the generator command and seed, and replay
the same saved trace for every policy comparison. Actual cache behavior depends
on tokenization and cache block boundaries.
