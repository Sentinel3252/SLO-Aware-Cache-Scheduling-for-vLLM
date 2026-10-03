# Design rationale

CacheFlow changes which waiting request a single vLLM V1 engine tries to admit
next. Keeping ranking separate from allocation makes the policies easier to test;
vLLM still handles execution and the request lifecycle.

## When ordering may matter

A long prompt with no cached prefix may need much more prefill work than a
continuation whose prefix is already cached. FCFS uses arrival order without
considering that difference. Chunked prefill limits work per iteration, but the
order in which requests enter the batch can still affect TTFT and prefix reuse.

Reordering is more likely to help when the queue is busy and its first K requests
have different prefill costs or reusable cached prefixes. At low load, or with
similar prompts, there may be little to gain. Decode contention and resource
limits can also make a policy slower than FCFS.

## Tradeoffs

- **Cache probes:** normal allocation lookup paths can affect metrics/events.
  Candidate extraction uses guards and coordinator lookup without allocation.
- **Residual cost:** cached tokens already reduce residual work; extra cache credit
  defaults to zero to avoid rewarding the same benefit twice.
- **Context cost:** new suffix queries still attend to cached context. The optional
  correction needs model/GPU calibration and does not replace measured timing.
- **SLO ordering:** inverse slack becomes unstable near zero and collapses overdue
  requests. Feasibility-first ordering is a heuristic, with an age override.
- **Fairness:** age protection acts only on visible candidates and cannot promise
  a wall-clock wait bound under queue/resource pressure.
- **Integration:** private upstream interfaces require pinned targets and separate
  source-contract and GPU validation.

## Related upstream discussions

The following discussions informed the design. They do not imply upstream
endorsement or an agreed interface:

- [Scheduler Plugin RFC #51608](https://github.com/vllm-project/vllm/issues/51608)
- [Cache-aware waiting QoS #53277](https://github.com/vllm-project/vllm/issues/53277)
- [Prefix-cache scheduling #7883](https://github.com/vllm-project/vllm/issues/7883)
- [Cache-affinity RFC #42185](https://github.com/vllm-project/vllm/issues/42185)
- [Tagged v0.28.0 scheduler](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/v1/core/sched/scheduler.py)

The [recorded results](results.md) show how some of these choices behaved in GPU
tests. Further workload tests are needed; the simulator alone cannot tell us
whether serving latency improves.
