# Contributing

See `CONTRIBUTING.md` for full details.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Build docs locally

```bash
pip install -e ".[docs]"
mkdocs serve
```

