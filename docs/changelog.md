# Changelog

## Unreleased

- Started the 0.2.0 automation-ready triage plan.
- Added `mlenvdoctor doctor --json` for compact machine-readable triage summaries.
- Aligned MCP `doctor_summary` responses with the shared doctor summary payload.

## 0.1.6

- Restored Python 3.8 import compatibility for annotation-heavy modules.
- Declared the direct Click runtime dependency required by the CLI entry point.
- Fixed homepage action buttons so they navigate to deployed `.html` pages instead of 404 routes.
- Improved homepage button dimensions, spacing, hover, and focus states.
- Trimmed source distribution file selection to project sources, tests, docs, and release metadata.

## 0.1.5

- Added guided doctor summaries for beginner-friendly recovery.
- Improved report generation with JSON and HTML evidence.
- Expanded root-cause grouping for PyTorch, CUDA, TensorFlow, JAX, Docker, and dependency stacks.
- Added safer fix planning, verification, and rollback-aware file generation.

## 0.1.x

- Added `diagnose`, `doctor`, `report`, `fix`, `dockerize`, `stack`, and MCP commands.
- Added export formats for CI and teammate handoff.
- Added tests for CLI, validators, Docker generation, fixes, and utilities.

## Roadmap

- More framework compatibility matrices.
- Richer Pipenv and Conda stale-lock detection.
- Deeper container and WSL2 evidence collection.
- First-class check plugin interface.
