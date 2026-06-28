# Dependency Maintenance

This project keeps runtime dependency ranges in `pyproject.toml` and uses `uv.lock` as the reproducible package update plan for contributors.

## Current Refresh

The dependency refresh executed on 2026-06-28 did three things:

- upgraded `uv.lock` with `uv lock --upgrade`
- updated pre-commit hook pins for Black, Ruff, mypy, and standard repository checks
- kept documentation dependencies sourced from the `docs` optional extra instead of duplicating them in `requirements-docs.txt`

The lockfile may contain multiple versions of the same tool because the project supports Python 3.8 and newer. That is expected when upstream packages publish different compatibility windows.

## Routine Update Plan

1. Refresh the lockfile:

   ```bash
   uv lock --upgrade
   ```

2. Refresh pre-commit hook pins:

   ```bash
   python -m pre_commit autoupdate
   ```

3. Validate the lockfile and project checks:

   ```bash
   uv lock --check
   python -m black --check src tests
   python -m ruff check src tests
   python -m pytest
   python -m pip install -r requirements-docs.txt
   mkdocs build --strict
   ```

The pre-commit mypy hook is manual because CI treats mypy as advisory today. Run it with `python -m pre_commit run mypy --hook-stage manual --all-files` when working specifically on type coverage.

## Cleanup Policy

Generated diagnostic exports are local artifacts. Do not commit files such as `mlenvdoctor-report/`, `results.json`, `results.html`, `line.json`, or `line.csv`; regenerate them from the CLI when needed.
