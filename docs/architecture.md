# Architecture

## Scheduling boundary

On each `CARASScheduler.schedule()` call, the adapter takes the first K requests
from the baseline waiting queue and ranks them by cache hits, remaining work, age,
and SLO targets. It then passes that view to vLLM's scheduler. Requests stay in
the baseline container until removed or popped, so each cycle uses current state.

The adapter does not change running-request scheduling, token budgets, chunked
prefill, KV allocation, preemption decisions, or request lifecycle. If upstream
cannot allocate resources for the selected request and stops admission, CacheFlow
does not search the remaining queue for a request that fits.

## Components

| Module | Responsibility |
| --- | --- |
| `cacheflow.config` | Immutable policy configuration and environment parsing |
| `cacheflow.features` | Request features and guarded local prefix lookup |
| `cacheflow.cost_model` | Deterministic ranking keys |
| `cacheflow.request_queue` | Bounded ranked view and upstream queue operations |
| `cacheflow.vllm_adapter` | Startup capability checks and schedule delegation |
| `cacheflow.metrics` | Cumulative reorder telemetry |
| `cacheflow.tools` | Installed compatibility, output, and policy comparison tools |
| `benchmark` | Traces, streaming metrics, ordering simulation, and reports |

## Queue contract

FCFS stores baseline requests in an ordered mapping. Preempted requests prepended
under the default protection setting live in a separate front deque and are not
re-ranked. In priority mode, a heap with lazy deletion preserves upstream priority
ordering; priority remains an outer key during policy ranking.

Every visible candidate is scored. The ranked view never promotes a request from
beyond its window. Equal keys use arrival time and request ID. The age override
applies within the visible window and priority class, not across all requests.
See [configuration](configuration.md) for mode definitions.

## Cache feature extraction

The provider applies version-specific eligibility guards before querying the local
coordinator's `find_longest_cache_hit`. It avoids using `get_computed_blocks` as a
candidate probe because that path can record cache statistics or events. The
probe must not touch blocks, reference counts, or free queues.

The lookup return format and eligibility guards are private APIs. CPU fixtures
cover them, and integration checks compare them with the target vLLM source.
The provider does not query remote KV connectors. Residual tokens and context
correction are estimates; vLLM still decides how to allocate KV space and execute.

## Complexity and telemetry

Scoring and sorting are bounded by K, with `O(K log K)` sorting. Cache-probe cost
also depends on prefix length and coordinator internals. FCFS candidate enumeration
visits K entries; priority enumeration uses heap operations normally costing
`O(K log N)`, with additional work when stale entries are cleaned up. Iterating the
whole queue and bulk removals can still visit N requests.

Telemetry measures `prepare()` elapsed time, not all adapter overhead: snapshot
recording and optional synchronous JSONL writes happen afterward. Evaluate those
costs separately. See [telemetry limits](configuration.md#telemetry).

## Compatibility

The adapter checks required scheduler/cache attributes and forwards schedule
arguments using `*args, **kwargs`. Future signature changes may still break it.
The [compatibility matrix](compatibility.md) lists which versions have source
review, mocked tests, static checks, or GPU measurements.
