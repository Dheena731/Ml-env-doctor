# Changelog

All notable changes to ML Environment Doctor are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.3] - 2026-04-09

### Changed
- Refactored diagnostics into deeper framework-specific checks for Python, PyTorch, TensorFlow/Keras, and JAX/Flax
- Expanded machine-readable export payloads with `check_id`, `category`, `confidence`, `evidence`, and metadata
- Reworked `fix` into a structured plan/apply/dry-run workflow with verification after successful changes
- Made Dockerfile generation more configurable with stack, base image, and Python version selection
- Replaced install-dependent CLI subprocess tests with Typer `CliRunner` tests
- Updated project documentation to distinguish current behavior from roadmap items

### Added
- Python runtime compatibility check
- PyTorch CUDA execution probe
- TensorFlow execution probe
- JAX execution probe
- `mlenvdoctor fix --apply`
- `mlenvdoctor fix --yes`
- `mlenvdoctor fix --dry-run`
- richer CSV and HTML exports

## [0.1.2] - 2026-04-08

### Added
- JSON/HTML report generation
- `doctor --ci`
- MCP JSON-lines stub

## [0.1.0] - 2024-XX-XX

### Added
- Initial release of ML Environment Doctor
