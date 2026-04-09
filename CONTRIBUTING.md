# Contributing to ML Environment Doctor

Thank you for your interest in contributing to ML Environment Doctor! 🎉

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/ml_env_doctor.git`
3. Create a branch: `git checkout -b feature/your-feature-name`
4. Install in development mode: `pip install -e ".[dev]"`
5. Make your changes
6. Run tests: `pytest`
7. Run linters: `black src/ tests/ && ruff check src/ tests/`
8. Commit your changes: `git commit -m "Add feature: your feature"`
9. Push to your fork: `git push origin feature/your-feature-name`
10. Open a Pull Request

## Development Setup

```bash
# Install dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run with coverage
pytest --cov=mlenvdoctor --cov-report=html
```

## Code Style

We use:
- **Black** for code formatting (line length: 100)
- **Ruff** for linting
- **mypy** for type checking (optional for now)

```bash
# Format code
black src/ tests/

# Check linting
ruff check src/ tests/

# Type check
mypy src/
```

## Testing

- Write tests for new features
- Ensure all tests pass: `pytest`
- Aim for good test coverage
- Tests are in the `tests/` directory
- Prefer Typer `CliRunner` for CLI behavior tests so they do not depend on the shell command being installed
- Reserve subprocess-based tests for packaging or integration scenarios only

## Commit Messages

Use clear, descriptive commit messages:
- `Add feature: GPU memory check`
- `Fix: CUDA version detection on Windows`
- `Update: Documentation for dockerize command`

## Pull Request Process

1. Update README.md if needed
2. Update CHANGELOG.md with your changes
3. Ensure all tests pass
4. Ensure code is formatted and linted
5. Request review from maintainers

## Project Structure

```
mlenvdoctor/
├── src/mlenvdoctor/     # Source code
│   ├── cli.py          # CLI entrypoint
│   ├── diagnose.py     # Diagnostic logic
│   ├── fix.py          # Auto-fix logic
│   ├── dockerize.py    # Dockerfile generation
│   ├── gpu.py          # GPU tests
│   └── utils.py        # Utilities
├── tests/              # Test suite
└── docs/               # Documentation (if added)
```

## Areas for Contribution

- 🐛 Bug fixes
- ✨ New features (diagnostics, fixes, Docker templates)
- 📚 Documentation improvements
- 🧪 Additional tests
- 🎨 UI/UX improvements
- ⚡ Performance optimizations
- 🌍 Support for additional ML frameworks

## Questions?

Open an issue for questions or discussions. We're happy to help!

Thank you for contributing! 🙏
