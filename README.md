# ML Environment Doctor

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/pypi/v/mlenvdoctor.svg)](https://pypi.org/project/mlenvdoctor/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/mlenvdoctor?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/mlenvdoctor)
[![Repository](https://img.shields.io/badge/GitHub-Ml--env--doctor-black?logo=github)](https://github.com/Dheena731/Ml-env-doctor)

> Single command fixes many "why is my ML environment broken?" problems.

ML Environment Doctor is a Python CLI for diagnosing and repairing ML environments. It checks CUDA, PyTorch, TensorFlow/Keras, and JAX/Flax readiness, generates shareable reports, emits CI-friendly exit codes, and creates Dockerfile templates for training workflows.

- Repository: `https://github.com/Dheena731/Ml-env-doctor`
- PyPI: `https://pypi.org/project/mlenvdoctor/`

## Why ML Environment Doctor?

**Problem**: ML environment setup is fragmented across forum answers, conflicting CUDA wheels, and half-working virtual environments.

**Solution**: one CLI that:

- Diagnoses CUDA, Python, and ML framework issues quickly
- Suggests copy-paste fixes for common failures
- Emits machine-readable JSON and stable exit codes for CI
- Saves HTML and JSON reports for teammates
- Generates recommended dependency stacks for LLM training

## Quick Start

```bash
# Install
pip install mlenvdoctor

# Human-readable diagnosis
mlenvdoctor diagnose

# Full scan
mlenvdoctor diagnose --full

# JSON to stdout for CI/parsers
mlenvdoctor diagnose --json -

# CI-friendly compact summary
mlenvdoctor doctor --ci

# Save a shareable report bundle
mlenvdoctor report

# Generate a recommended LLM training requirements file
mlenvdoctor stack llm-training -o requirements-llm-training.txt
```

## 30-Second Quickstart

```bash
pip install mlenvdoctor
mlenvdoctor diagnose
mlenvdoctor diagnose --json -
mlenvdoctor report
```

If you are debugging a teammate's machine or a CI runner, start with `mlenvdoctor report` and share the generated JSON/HTML pair.

## Features

### Diagnosis

- CUDA driver and GPU visibility checks
- PyTorch CUDA availability and version checks
- TensorFlow runtime and Keras 3 readiness checks
- JAX backend and Flax installation checks
- GPU memory warnings
- Disk space checks for model cache usage
- Docker GPU support detection
- Hugging Face connectivity checks

### Machine-Readable Output

- `mlenvdoctor diagnose --json -` prints JSON to stdout
- Exit codes are stable for automation:
  - `0`: healthy
  - `1`: warnings present
  - `2`: critical issues present

### Reports

- `mlenvdoctor report` saves timestamped JSON and HTML reports
- Output can be attached to CI jobs or shared with teammates

### Fixes

- Failures include copy-paste fix commands where possible
- `mlenvdoctor fix` can generate `requirements.txt` or Conda environment files
- `mlenvdoctor fix --venv` can create and use a virtual environment

### Stacks

- `mlenvdoctor stack llm-training` prints a recommended dependency stack for fine-tuning
- `mlenvdoctor fix --stack llm-training` uses that stack for generated requirements

### MCP

- `mlenvdoctor mcp serve` exposes a minimal JSON-line MCP stub
- Current stub tools:
  - `diagnose`
  - `get_fixes`

## Examples

### Diagnose

```bash
mlenvdoctor diagnose
mlenvdoctor diagnose --full
mlenvdoctor diagnose --json -
```

### CI

```bash
mlenvdoctor doctor --ci
mlenvdoctor doctor --ci --full
```

### Reports

```bash
mlenvdoctor report
mlenvdoctor report --quick --output-dir artifacts/mlenvdoctor
```

### Stacks

```bash
mlenvdoctor stack llm-training
mlenvdoctor stack llm-training -o requirements-llm-training.txt
```

### Auto-Fix

```bash
mlenvdoctor fix
mlenvdoctor fix --conda
mlenvdoctor fix --venv
mlenvdoctor fix --stack llm-training
```

### Dockerize

```bash
mlenvdoctor dockerize mistral-7b
mlenvdoctor dockerize --service
```

## Troubleshooting Examples

```bash
# Your CI job wants JSON
mlenvdoctor diagnose --json -

# Your team wants a report bundle
mlenvdoctor report --output-dir artifacts/mlenvdoctor

# Your pipeline wants a one-line summary plus fix hints
mlenvdoctor doctor --ci

# You want a starting dependency set for fine-tuning
mlenvdoctor stack llm-training -o requirements-llm-training.txt
```

## Installation

```bash
# From PyPI
pip install mlenvdoctor

# From source
git clone https://github.com/Dheena731/Ml-env-doctor.git
cd Ml-env-doctor
pip install -e .
```

## Development

```bash
git clone https://github.com/Dheena731/Ml-env-doctor.git
cd Ml-env-doctor
pip install -e ".[dev]"
pytest
ruff check src/ tests/
black src/ tests/
mypy src/
```

## Repository

- GitHub: [Dheena731/Ml-env-doctor](https://github.com/Dheena731/Ml-env-doctor)
- Issues: [Open an issue](https://github.com/Dheena731/Ml-env-doctor/issues)
- Clone URL: `https://github.com/Dheena731/Ml-env-doctor.git`

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
