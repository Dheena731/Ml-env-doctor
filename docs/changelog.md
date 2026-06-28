# Changelog

## Unreleased

- Refreshed the dependency lockfile and pre-commit hook pins.
- Reorganized the documentation navigation around user guides and reference pages.
- Removed tracked generated report/export artifacts and internal planning notes from the docs tree.

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
