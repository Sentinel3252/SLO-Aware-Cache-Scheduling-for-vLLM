# Compatibility and support

## Validation levels

| Surface | Target / environment | Evidence and limits |
| --- | --- | --- |
| Pure policy and CPU tools | Python >= 3.10 | Dependency-free unit tests; CI covers Linux and Windows on 3.10 and 3.12 |
| vLLM adapter | vLLM `0.28.0`, V1, synchronous scheduling | Dependency pin and source-reviewed interface target; no GPU validation of this target is shipped |
| Historical GPU run | `0.23.1rc1.dev818+gba2215209`, RTX 4090, Qwen2.5-3B | Recorded measurements only; raw artifacts are absent from this checkout |
| Older lookup shape | `0.23.x` two-value result and inline eligibility guard | Narrow CPU-tested compatibility branch; not a supported release range |
| Future vLLM releases | Unspecified | Require a separate compatibility check and integration validation |

The configured target is not a claim about the latest vLLM release. See the
[tagged upstream scheduler source](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/v1/core/sched/scheduler.py)
and [pinned dependency](../requirements/vllm.txt). The runtime needs Linux and the
accelerator/Python environment required by that vLLM build.

## What the checks cover

`cacheflow-check-vllm` imports the installed V1 classes and inspects a subset of
class-level scheduler, queue, and cache-eligibility interfaces. Adapter startup
also checks instance attributes and the coordinator lookup method. Neither check
verifies output correctness, scheduling behavior under load, or all signatures.

Mock-based CPU tests check delegation and both cache-lookup return shapes.
The dedicated CI source-contract job compares queue operations with the tagged
vLLM `0.28.0` source. Locally these tests are optional and require a source checkout.
Source-contract tests compare queue operations with a real upstream
`request_queue.py`, using a stub request class. They do not instantiate a GPU engine.

## Upgrade procedure

1. Record the exact installed vLLM version/commit and inspect the changed scheduler,
   queue, cache guards, and coordinator lookup semantics.
2. Run `cacheflow-check-vllm` and the CPU suite in the target environment.
3. Set `VLLM_SOURCE_TREE` to the matching upstream checkout and run
   `python -m unittest tests.test_upstream_contract -v`.
4. Launch the server, compare deterministic outputs against upstream FCFS, and
   exercise cancellation, preemption, cache hits/misses, and priority workloads.
5. Benchmark with equal server flags and report overhead and regressions.
6. Update the pin and this matrix only after recording the relevant evidence.

Keep all compared servers on `--no-async-scheduling`. Async scheduler parity,
distributed routing, remote KV feature queries, and arbitrary future interfaces
are outside the current validation scope.
