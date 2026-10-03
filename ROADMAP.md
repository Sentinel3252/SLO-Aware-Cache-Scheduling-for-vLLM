# Roadmap

This roadmap describes proposed work, not delivery dates or support guarantees.
Open an issue to discuss or take ownership of an item.

| Area | Next step | Acceptance criteria |
| --- | --- | --- |
| vLLM integration | Validate the pinned target on GPU | Exact version, reproducible flags, output comparison, failures, and source-contract results recorded |
| Reproducibility | Publish a sanitized benchmark artifact bundle | Trace, generator/seed, raw request metrics, model revision, environment, and checksums available together |
| Workload coverage | Evaluate distinct-prefix cache contention | Mixed, RAG, multi-session, and cold workloads; multiple loads and at least three seeds |
| Fairness | Add per-request waiting instrumentation | Max/p99 wait and overtaking distribution reported alongside goodput |
| Cost model | Calibrate context-sensitive prefill estimates | Predictions compared with measured timing across cached-context and suffix lengths |
| Observability | Improve metric export lifecycle | Process-safe output and final partial-interval handling validated with overhead measurements |
| Upstream integration | Track scheduler extension APIs | Migration considered when a suitable stable upstream interface exists |

The current scope remains waiting admission within one V1 engine. Cross-worker
routing, remote KV retrieval, and eviction policies require separate designs.
