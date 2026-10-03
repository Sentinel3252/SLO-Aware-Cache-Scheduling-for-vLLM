# Changelog

Changes are recorded here for users and contributors. This file does not imply
that a tag or package has been published.

## Unreleased

### Added

- Trace format reference and a hand-written CPU replay example.
- Regression coverage for environment configuration, report denominators and
  duration, cache lookup guards, metric snapshots, and cumulative file exports.
- English and Chinese entry points, installation guide, configuration reference,
  compatibility matrix, and contributor documentation.
- Packaged compatibility, output-comparison, policy-comparison, and study commands.
- Issue forms, pull request template, community guidelines, and release checklist.
- Wheel smoke validation outside the checkout and source archive content checks.

### Changed

- Documented historical GPU data availability and distinguished an integration
  target from an end-to-end validated runtime.
- Removed interview-oriented positioning from public technical documentation.
- Centralized package version metadata and excluded personal material from archives.

### Fixed

- Reject non-finite floating-point policy parameters at startup.
- Reject empty output captures/comparisons and nonpositive capture limits.
- Validate generator dimensions, mixed lengths, finite QPS and SLO targets, and
  benchmark concurrency/timeout before performing work.
- Keep RAG document prefixes distinct beyond twenty groups. Generated prompt text
  changes; record the version/commit and reuse saved traces across comparisons.

## 0.1.0 — baseline

- Dependency-free policy layer, bounded request queue, and vLLM V1 adapter.
- FCFS, cache, residual, aging, CARAS, and SLO policy modes.
- Synthetic trace generation, streaming benchmarking, simulation, and reporting.
- Scheduler aggregate telemetry and CPU correctness tests.
