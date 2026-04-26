# ML Environment Doctor

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/pypi/v/mlenvdoctor.svg)](https://pypi.org/project/mlenvdoctor/)
[![CI](https://github.com/Dheena731/Ml-env-doctor/actions/workflows/ci.yml/badge.svg)](https://github.com/Dheena731/Ml-env-doctor/actions/workflows/ci.yml)
[![PyPI - Downloads](https://static.pepy.tech/badge/mlenvdoctor)](https://pepy.tech/project/mlenvdoctor)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mlenvdoctor)](https://pypi.org/project/mlenvdoctor/)
[![codecov](https://codecov.io/gh/Dheena731/Ml-env-doctor/branch/master/graph/badge.svg)](https://codecov.io/gh/Dheena731/Ml-env-doctor)

> Diagnose ML environment issues, export shareable reports, and generate safe starting fixes.

ML Environment Doctor is a Python CLI for inspecting machine learning environments used for local development, CI jobs, and GPU containers. It focuses on practical readiness checks for PyTorch, TensorFlow/Keras, and JAX/Flax workflows, then turns those results into readable reports and automatable outputs.

Repository: `https://github.com/Dheena731/Ml-env-doctor`

## What It Does Today

- Diagnoses Python runtime compatibility
- Checks NVIDIA driver visibility with `nvidia-smi`
- Validates PyTorch version, CUDA visibility, and a basic CUDA tensor execution path
- Validates TensorFlow import, GPU enumeration, and a small tensor execution path
- Validates JAX backend visibility, Flax presence, and a small JAX array execution path
- Checks common ML training libraries like `transformers`, `peft`, `trl`, `datasets`, and `accelerate`
- Optionally checks GPU memory, disk space, Docker GPU support, and Hugging Face connectivity
- Exports JSON, CSV, and HTML reports
- Generates requirements files,  Conda environment files, and Dockerfiles
- Provides a safe fix workflow with planning, dry-run, apply, and verification modes

## What It Does Not Claim Yet

- It does not fully repair every environment problem automatically
- It does not maintain a large model registry
- It does not replace framework-native setup guides for CUDA, cuDNN, or TPU drivers
- Its Docker generation is configurable, but still template-based rather than a full build system

## Quick Start

```bash
pip install mlenvdoctor

mlenvdoctor diagnose
mlenvdoctor diagnose --full
mlenvdoctor diagnose --json -
mlenvdoctor report
mlenvdoctor fix --plan
mlenvdoctor fix --dry-run
mlenvdoctor fix --apply --yes
mlenvdoctor fix --verify
mlenvdoctor dockerize tinyllama --stack minimal
```

## Main Commands

### Diagnose

Detailed evidence and export command.

```bash
mlenvdoctor diagnose
mlenvdoctor diagnose --full
mlenvdoctor diagnose --json diagnostics.json
mlenvdoctor diagnose --json -
mlenvdoctor diagnose --csv diagnostics.csv --html diagnostics.html
```

`diagnose` returns stable exit codes:

- `0`: no warnings or critical issues
- `1`: warnings present
- `2`: critical issues present

### Doctor

Action-oriented triage command. It is intentionally different from `diagnose`:

- `doctor` tells you what failed, the likely cause, the best next fix, and how to verify it
- `diagnose` shows the full evidence and supports exports

Compact CI-friendly output:

```bash
mlenvdoctor doctor --ci
mlenvdoctor doctor --ci --full
mlenvdoctor doctor --guided
```

### Report

Save a shareable JSON and HTML bundle:

```bash
mlenvdoctor report
mlenvdoctor report --quick --output-dir artifacts/mlenvdoctor
```

### Fix

The fix workflow is intentionally explicit:

```bash
mlenvdoctor fix --plan
mlenvdoctor fix --dry-run
mlenvdoctor fix --apply
mlenvdoctor fix --apply --yes
mlenvdoctor fix --verify
mlenvdoctor fix --rollback
mlenvdoctor fix --venv --apply --yes
mlenvdoctor fix --conda
mlenvdoctor fix --stack llm-training --dry-run
```

Current fix behavior:

- Plans file-generation and environment actions from detected issues
- Labels each planned action by risk level
- Supports dry-run mode before changes
- Can create a virtual environment
- Can generate either `requirements-mlenvdoctor.txt` or `environment-mlenvdoctor.yml`
- Can install requirements when `--apply` is used
- Supports explicit verification via `mlenvdoctor fix --verify`
- Re-runs diagnostics for verification after a successful apply

### Stack

```bash
mlenvdoctor stack llm-training
mlenvdoctor stack llm-training --output requirements-llm-training.txt
```

### Dockerize

```bash
mlenvdoctor dockerize tinyllama
mlenvdoctor dockerize mistral-7b --stack llm-training
mlenvdoctor dockerize --service --base-image nvidia/cuda:12.4.0-runtime-ubuntu22.04
mlenvdoctor dockerize gpt2 --python-version 3.10 -o Dockerfile.gpt2
```

Current Docker generation supports:

- model-aware defaults for `tinyllama`, `gpt2`, and `mistral-7b`
- stack selection
- base image override
- Python version selection
- training and FastAPI service profiles

### MCP

```bash
mlenvdoctor mcp serve
```

MCP now exposes a JSON-lines interface over stdin/stdout.

Core integration tools:

- `diagnose`
- `get_fixes`
- `doctor_summary`
- `report_bundle`
- `diagnose_environment` (v1 alias)
- `get_fix_plan` (v1 alias)
- `verify_fix` (v1 alias)
- `export_report` (v1 alias)
- `list_tools`
- `tool_schema`
- `health`

Request format:

```json
{"tool":"list_tools","arguments":{}}
```

```json
{"tool":"tool_schema","arguments":{"name":"report_bundle"}}
```

Compatibility policy:

- MCP schemas evolve additively.
- v1 alias names are kept for migration safety.
- Use `tool_schema` to discover canonical tool names.

## Output Schema

JSON exports include:

- issue name, status, severity, and fix command
- check ID and category
- recommendation, likely cause, and verify steps
- confidence
- evidence and metadata blocks
- runtime context (platform, detected backend, NVIDIA tooling visibility)
- summary counts and exit code

That makes the tool more useful for CI parsing, dashboards, and future integrations.

## Development

```bash
git clone https://github.com/Dheena731/Ml-env-doctor.git
cd Ml-env-doctor
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m black src tests
python -m ruff check src tests
python -m pytest
python -m mypy src
```

## CI/CD and Release Flow

- CI runs formatting (`black --check`), lint (`ruff`), tests (`pytest`), and package build checks (`python -m build`, `twine check`) on every push/PR.
- The matrix tests Linux (3.8-3.11), macOS (3.11), and Windows (3.11).
- A tagged GitHub release triggers the publish job to upload built artifacts to PyPI.
- If CI fails locally vs. GitHub, run these exact commands first:

```bash
python -m black src tests
python -m ruff check src tests
python -m pytest
```

## Test Strategy

The repository uses two test layers:

- direct Python and Typer `CliRunner` tests for most command behavior
- a smaller set of module-level tests for exports, validators, diagnostics, and file generation

This keeps the main test suite independent from whether `mlenvdoctor` was installed as a shell command in the current environment.

## Repository Notes

- [CONTRIBUTING.md](CONTRIBUTING.md) covers local setup and contribution flow
- [IMPROVEMENTS.md](IMPROVEMENTS.md) tracks active improvement themes
- [IMPROVEMENTS_ROADMAP.md](IMPROVEMENTS_ROADMAP.md) outlines future milestones
- [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md) explains the current project architecture and product shape
- [docs/FAILURE_TAXONOMY.md](docs/FAILURE_TAXONOMY.md) defines prioritized mismatch codes used by diagnostics and MCP
- [docs/KILLER_PRODUCT_PLAN.md](docs/KILLER_PRODUCT_PLAN.md) is the strategic roadmap for turning the project into a category-leading product
- [docs/PHASED_EXECUTION_PLAN.md](docs/PHASED_EXECUTION_PLAN.md) breaks the strategy into execution phases and current build focus
- [docs/CASE_STUDY_TEMPLATE.md](docs/CASE_STUDY_TEMPLATE.md) helps publish real failure-to-fix stories for user onboarding
- [docker/README.md](docker/README.md) documents Docker-specific workflows

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
