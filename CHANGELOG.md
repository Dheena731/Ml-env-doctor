# Changelog

All notable changes to ML Environment Doctor will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-XX-XX

### Added
- Initial release of ML Environment Doctor
- `diagnose` command for environment diagnostics
  - CUDA driver detection
  - PyTorch/CUDA compatibility checks
  - ML library version checks (transformers, peft, trl, datasets, accelerate)
  - GPU memory checks (with `--full` flag)
  - Disk space checks
  - Docker GPU support detection
  - Internet connectivity checks
- `fix` command for auto-fixing environment issues
  - Requirements.txt generation
  - Conda environment file generation
  - Virtual environment creation
  - Automatic dependency installation
- `dockerize` command for Dockerfile generation
  - Model-specific Dockerfiles (mistral-7b, tinyllama, gpt2)
  - FastAPI service template generation
  - CUDA 12.4 base images
- `test-model` command for model smoke tests
- `smoke-test` command for LoRA fine-tuning smoke tests
- Rich UI with colored output and tables
- Comprehensive test suite
- CI/CD workflow with GitHub Actions
- Documentation (README.md, CONTRIBUTING.md)

[0.1.0]: https://github.com/yourusername/ml_env_doctor/releases/tag/v0.1.0

