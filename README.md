# 🔍 ML Environment Doctor

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/pypi/v/mlenvdoctor.svg)](https://pypi.org/project/mlenvdoctor/)

> **Single command fixes 90% of "my torch.cuda.is_available() is False" issues.**

ML Environment Doctor is a production-ready Python CLI that diagnoses, auto-fixes, and Dockerizes ML environments for LLM fine-tuning. It detects CUDA conflicts, generates locked requirements.txt/conda envs, tests GPU readiness with real LLM smoke tests, and outputs production Dockerfiles.

## 🎯 Why ML Environment Doctor?

**Problem**: LLM fine-tuning setup is fragmented across StackOverflow answers, conflicting PyTorch/CUDA versions, and missing dependencies. Hours wasted debugging `torch.cuda.is_available() == False`.

**Solution**: ONE TOOL that:
- ✅ Diagnoses your environment in <5 seconds
- ✅ Auto-fixes 80% of common issues
- ✅ Generates production-ready Dockerfiles
- ✅ Tests GPU readiness with real models
- ✅ Supports PyTorch 2.4+ with CUDA 12.1/12.4

## 🚀 Quick Start

```bash
# Install
pip install mlenvdoctor

# Diagnose your environment
mlenvdoctor diagnose

# Full diagnostic scan
mlenvdoctor diagnose --full

# Auto-fix issues and generate requirements.txt
mlenvdoctor fix

# Generate Dockerfile for fine-tuning
mlenvdoctor dockerize mistral-7b

# Run smoke test with real model
mlenvdoctor test-model tinyllama
```

## 📋 Features

### 🔍 Diagnosis

- **CUDA Detection**: NVIDIA driver, CUDA version, GPU availability
- **PyTorch/CUDA Compatibility**: Version matrix matching
- **Library Checks**: transformers, peft, trl, datasets, accelerate
- **GPU Memory**: Available memory for fine-tuning
- **Disk Space**: HF cache space warnings (~50GB)
- **Docker GPU**: nvidia-docker support detection
- **Connectivity**: Hugging Face Hub access

```bash
mlenvdoctor diagnose --full
```

Output:
```
🔍 Running ML Environment Diagnostics...

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ ML Environment Doctor - Diagnostic Results                                           ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Issue              │ Status                        │ Severity │ Fix                    │
├────────────────────┼───────────────────────────────┼──────────┼────────────────────────┤
│ NVIDIA GPU Driver  │ ✅ PASS - CUDA 12.4          │ INFO     │                        │
│ PyTorch CUDA       │ ✅ PASS - CUDA 12.4 (1 GPU)  │ INFO     │                        │
│ transformers       │ ✅ PASS - 4.44.0              │ INFO     │                        │
│ peft               │ ❌ FAIL - Not installed       │ CRITICAL │ pip install peft>=0.12 │
└────────────────────┴───────────────────────────────┴──────────┴────────────────────────┘

✅ Passed: 3
❌ Critical Issues: 1
```

### 🔧 Auto-Fix

Generates optimized `requirements.txt` or conda environment files based on detected issues.

```bash
# Generate requirements.txt
mlenvdoctor fix

# Generate conda environment
mlenvdoctor fix --conda

# Create virtual environment and install
mlenvdoctor fix --venv
```

### 🐳 Dockerize

Generate production-ready Dockerfiles for ML fine-tuning.

```bash
# Basic Dockerfile
mlenvdoctor dockerize mistral-7b

# FastAPI service template
mlenvdoctor dockerize --service

# Custom output
mlenvdoctor dockerize tinyllama -o Dockerfile
```

Generated Dockerfile includes:
- NVIDIA CUDA 12.4 base image
- PyTorch with CUDA support
- ML libraries (transformers, peft, trl, accelerate)
- Optimized layer caching
- GPU runtime configuration

### 🧪 Testing

Run smoke tests with real LLM models to verify fine-tuning readiness.

```bash
# Test with TinyLlama (fast)
mlenvdoctor test-model tinyllama

# Test with Mistral-7B (requires 16GB+ GPU)
mlenvdoctor test-model mistral-7b

# LoRA smoke test
mlenvdoctor smoke-test
```

## 📦 Installation

```bash
# From PyPI (when published)
pip install mlenvdoctor

# From source
git clone https://github.com/dheena731/ml_env_doctor.git
cd ml_env_doctor
pip install -e .
```

## 🛠️ Development

```bash
# Clone repository
git clone https://github.com/dheena731/ml_env_doctor.git
cd ml_env_doctor

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linters
black src/ tests/
ruff check src/ tests/
mypy src/

# Pre-commit hooks
pre-commit install
```

## 📚 CLI Reference

### `diagnose`

Diagnose ML environment issues.

```bash
mlenvdoctor diagnose           # Quick scan
mlenvdoctor diagnose --full    # Full scan with GPU benchmark
```

### `fix`

Auto-fix environment issues.

```bash
mlenvdoctor fix                # Generate requirements.txt
mlenvdoctor fix --conda        # Generate conda environment
mlenvdoctor fix --venv         # Create virtual environment
mlenvdoctor fix --stack minimal # Use minimal ML stack
```

### `dockerize`

Generate Dockerfile for ML fine-tuning.

```bash
mlenvdoctor dockerize [model]              # Model: mistral-7b, tinyllama, gpt2
mlenvdoctor dockerize --service            # FastAPI service template
mlenvdoctor dockerize mistral-7b -o Dockerfile
```

### `test-model`

Test model loading and forward pass.

```bash
mlenvdoctor test-model tinyllama
mlenvdoctor test-model mistral-7b
mlenvdoctor test-model gpt2
```

### `smoke-test`

Run LoRA fine-tuning smoke test.

```bash
mlenvdoctor smoke-test
```

## 🎯 Use Cases

1. **Fresh Setup**: Diagnose and fix a new ML environment in minutes
2. **CUDA Issues**: Detect and fix PyTorch/CUDA version mismatches
3. **Production Deployment**: Generate Dockerfiles for containerized training
4. **CI/CD**: Verify GPU readiness in automated pipelines
5. **Environment Debugging**: Quick diagnosis of "why is my GPU not working?"

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Typer](https://typer.tiangolo.com/) and [Rich](https://rich.readthedocs.io/)
- Inspired by the LLM fine-tuning community's setup struggles
- Thanks to Hugging Face for amazing ML libraries

## ⭐ Star History

If you find this tool helpful, please star the repository! Our goal: **500 GitHub stars in the first month**.

---

**Made with ❤️ for the ML fine-tuning community**

