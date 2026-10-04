# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Evaluation framework (`src/precision_health_os/evaluation/`) with structured
  benchmark runners and regression tracking.
- Benchmark suite (`src/precision_health_os/benchmarks.py`) covering all seven
  NP-hard solver classes; all benchmarks passing.
- Coverage wave: 373 new tests across every module (143 → 516 total).
- Per-module gap tests: `test_*_gaps.py` for api, cli, clinical, drug_discovery,
  genomics, integration, iot, ml, optimization, security.
- `tests/evaluation/` — evaluation-specific test coverage.
- `tests/iot/test_ewma_warmup.py` — dedicated EWMA warm-up false-positive tests.
- `web/dashboard.html` — single-file, accessible system dashboard (zero absolute
  positioning, semantic landmarks, `prefers-reduced-motion` fallback).
- `web/benchmarks.html` — benchmark results report with an accessible results
  table (`<caption>`, `<th scope>`, `<colgroup>`, keyboard-scrollable region).
- `src/precision_health_os/dashboard/` — JSON-serializable adapter exposing live
  platform state to the web dashboards (100 % covered, built test-first).
- `docs/evaluation.md`, `docs/ui.md` — evaluation framework and UI/accessibility
  contract references.

### Changed

- TSP 2-opt delta optimisation: gap reduced from 1.7 % to 1.1 %.
- `clinical/__init__.py` — dead-code removal in `evaluate_rules`.
- `iot/__init__.py` — EWMA warm-up fix to eliminate false positives on initial
  observations.
- `optimization/__init__.py` — 2-opt improvement and internal refactoring.
- `pyproject.toml` — updated dependencies and tool configuration.

### Fixed

- **IoT: `detect_iqr` crashed on a degenerate interquartile range** — a flat
  baseline with a single outlier (e.g. `[70, 70, 70, 70, 120]`) raised
  `ZeroDivisionError` at the score calculation, taking the monitoring pipeline
  down for exactly the excursion that mattered most. Degenerate ranges are now
  handled explicitly: values off the median are flagged, and the score is
  derived from the deviation instead of a division by zero.
- **API: `acknowledge_alert` never acknowledged the alert** — it logged an audit
  event and returned `True` unconditionally without calling the alert manager,
  so a cleared alert stayed in the active set and reappeared to the clinician.
  It now delegates to `AlertManager.acknowledge` and returns `False` for an
  unknown alert id rather than reporting success it did not deliver.
- EWMA warm-up false positives: the anomaly detector now correctly handles the
  initial observation window, preventing spurious alerts on the first few readings.
- **Test integrity: two audit tests asserted the buggy behaviour as correct** —
  one used `pytest.raises(ZeroDivisionError)` to document the crash, the other
  acknowledged a fabricated alert id and asserted `True`, matching the broken
  implementation. Both were rewritten to assert correct behaviour.

## [0.1.1] - 2026-10-04

### Fixed

- **CI: ruff/bandit findings** — resolved all lint and security warnings; added
  comprehensive README documentation.
- **CI: StrEnum (UP042)** — migrated string enums to `StrEnum` to satisfy ruff
  rule UP042; applied `ruff format` across the codebase.
- **CI: codecov token errors** — eliminated token-related upload failures; added
  bandit configuration; upgraded GitHub Actions versions.
- **CI: secrets context in step `if`** — replaced direct `secrets.*` references
  in step-level `if` conditions with environment-variable indirection to comply
  with GitHub Actions expression constraints.
- **Clinical: genotype-aware PGx scoring** — pharmacogenomic scoring now correctly
  accounts for patient genotype when computing CPIC guideline scores, fixing
  incorrect drug-gene interaction calls for patients with known variants.
- **Clinical: EWMA self-inflation** — the exponentially weighted moving average
  anomaly detector no longer inflates its own baseline when processing anomalous
  readings, preventing drift that would mask subsequent true anomalies.

## [0.1.0] - 2026-10-04

### Added

- Initial release of Precision Health OS.
- 10 research clusters covering 68 bottlenecks, 41 NP-hard problems, 57 OSS
  projects, and 67 cited papers.
- 7 Mermaid architecture diagrams (system overview, data flow, NP-hard problem
  map, multi-agent coordination, digital twin feedback loop, genomics pipeline,
  clinical decision support flow).
- 10 core modules: `optimization`, `clinical`, `ml`, `iot`, `genomics`,
  `drug_discovery`, `security`, `integration`, `api`, `models`.
- 143 tests across all modules.
- NP-hard solvers: VRPTW (Clarke-Wright), TSP (NN + 2-opt), Knapsack (DP),
  Bipartite Matching (greedy).
- HIPAA-grade security: AES-256-GCM encryption, hash-chained audit trail, RBAC,
  Safe Harbor de-identification.
- FHIR/HL7 interoperability layer.
- Typer-based CLI (`phos`).
- Docker Compose deployment (app + Postgres + Redis + worker).
