# Results

These GPU measurements were recorded on 2026-09-19. The original notes point to
per-request JSONL files under `results/gpu/`, but those files are **missing from
this checkout**, and the notes contain no public download link. The table keeps
the reported numbers; without the raw files, we cannot independently verify them.

## Environment

- Date: 2026-09-19
- vLLM: `0.23.1rc1.dev818+gba2215209` / `ba22152096`
- Driver / CUDA runtime: 595.71.05 / PyTorch CUDA 13.0
- GPU: NVIDIA GeForce RTX 4090, 24,564 MiB
- Model: Qwen2.5-3B-Instruct, FP16
- Common flags: prefix caching + chunked prefill, sync scheduler,
  `max-num-seqs=32`, `max-num-batched-tokens=4096`, 4,096 KV blocks
- CacheFlow: bounded window K=16; max-wait=500 ms unless noted

## Correctness

- Historical CPU suite: 29 tests discovered, 27 passed, 2 source-contract tests skipped unless
  `VLLM_SOURCE_TREE` is explicitly set
- vLLM compatibility check: passed
- End-to-end requests: all retained runs below completed with 0 failures
- The benchmark used temperature=0 and streaming interval 1. This does not by
  itself establish output equivalence or exact token counts.

## Main results

| Workload | Offered load | Policy | TTFT p50 ms | TTFT p95 ms | TTFT p99 ms | SLO attainment | SLO goodput | Throughput |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mixed 128/512/1024/2048 | 80 QPS | FCFS | 1187.2 | 1497.4 | 1564.6 | 20.4% @ 1s | 12.70 req/s | 62.2 req/s |
| Mixed 128/512/1024/2048 | 80 QPS | Residual K=16 | 1189.8 | 1502.7 | 1537.3 | 19.2% @ 1s | 11.84 req/s | 61.8 req/s |
| RAG, 8 prefix groups | 320 QPS | FCFS | 2247.1 | 3470.3 | 3583.5 | 0% @ 200ms | 0 | 61.4 req/s |
| RAG, 8 prefix groups | 320 QPS | Cache K=16 | 2300.7 | 3570.3 | 3665.5 | 0% @ 200ms | 0 | 59.7 req/s |

## Missing measurements

The retained table does not include reorder overhead, cold-workload regression,
per-request waiting distributions, or starvation observations. Aggregate scheduler
metrics do not show how long individual requests waited. Reproducing the comparison
will require the raw trace, request metrics, exact model revision, and runs with
multiple seeds. The pinned vLLM `0.28.0` target also needs its own GPU validation.

## Interpretation

Residual ordering reduced mixed-workload P99 by 1.7%, but did not improve the
median, P95, or 1-second SLO goodput. Cache-only ordering increased RAG P95 by 2.9%.
The original CARAS defaults (`aging_tokens_per_ms=2`, max-wait=2s) performed worse
in an exploratory comparison: P95 was 3114.7 ms, versus 840.5 ms for FCFS.

One possible explanation is that repeatedly favoring small residual work delays
other requests until they reach the max-wait threshold. Cache hit count may also
be a poor ranking signal when request lengths are similar and the working set is
thrashing. We lack per-request waiting data to check these explanations.

These runs used one GPU, one model, synthetic prompts, and one retained seed per
table entry. They are not enough to recommend enabling the policy by default.
Compare it with upstream FCFS on the intended workload, using more seeds.
