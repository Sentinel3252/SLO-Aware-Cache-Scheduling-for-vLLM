# Contributing to CacheFlow

CacheFlow welcomes bug fixes, compatibility updates, documentation, and
reproducible workload reports. Use English or Chinese in issues; keep primary
technical documentation in English and update the Chinese README when entry-point
behavior changes.

## Development setup

Fork the repository, clone your fork, and create a branch for the change.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m unittest discover -v
ruff check .
ruff format --check .
```

PowerShell activation is `.venv\Scripts\Activate.ps1`. CPU tests require no GPU
or vLLM. Two upstream source-contract tests are skipped unless `VLLM_SOURCE_TREE`
points to a vLLM checkout; see [compatibility](docs/compatibility.md).

## Choose the right change

- Bugs: include a small reproduction, expected/actual behavior, and versions.
- Integration: include the exact vLLM release or commit, flags, and affected API.
- Policies: explain the workload and tradeoff, preserve deterministic ties, and
  add a behavior test covering the changed ranking or queue contract.
- Performance: attach environment details, traces or a generator command, all
  measured runs, failures, overhead, and limitations. Label simulated results.
- Documentation: verify commands and relative links before opening the PR.

Check existing issues first. Discuss changes to scheduling boundaries or new
runtime dependencies in an issue before substantial implementation.

## Engineering boundaries

Keep pure policy code importable without vLLM. The adapter changes waiting admission
order only: delegate KV allocation, token budgets, preemption, running scheduling,
and lifecycle to upstream. Cache probes must not mutate cache state, counters, or
events. Preserve priority classes and the protected FCFS preemption path.

Features must stay bounded by the candidate window. Do not justify a change using
simulated GPU speedups. Changes affecting upstream interfaces need contract tests
and a compatibility record; do not silently broaden the supported version range.

## Validate the package

```bash
python -m build
python -m twine check dist/*
python scripts/check_distribution.py --dist-dir dist
```

The distribution check installs the wheel into a temporary environment and runs
all CLI help commands plus a trace/simulation/report workflow outside the checkout.
CI runs this check, CPU tests, lint, and a queue-contract job against tagged vLLM
source. GPU integration is a separate
validation level.

## Submit a pull request

Explain the problem, changed behavior, and checks run. Keep the PR focused and
update the relevant docs and [changelog](CHANGELOG.md). Include regression tests
for scheduling behavior changes; avoid tests that only copy implementation details.
Do not include generated traces, prompt logs, credentials, or personal preparation
material. Personal preparation material lives under ignored `.local/`; this and
any legacy `interview/` directory are excluded from release archives.

Contributions are under the existing [Apache-2.0 license](LICENSE). Follow the
[community guidelines](CODE_OF_CONDUCT.md); report security concerns through
[SECURITY.md](SECURITY.md).
