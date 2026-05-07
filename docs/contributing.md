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
python scripts/build_docs.py
python -m http.server 8000 -d docs_dist
```

