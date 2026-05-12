# Contributing

Contributions are welcome, especially new checks for real ML environment failures.

## Local Setup

```bash
git clone https://github.com/Dheena731/Ml-env-doctor.git
cd Ml-env-doctor
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pre-commit install
pytest
```

=== "Windows PowerShell"

    ```powershell
    .\.venv\Scripts\Activate.ps1
    python -m pip install -e ".[dev]"
    pytest
    ```

=== "macOS / Linux"

    ```bash
    source .venv/bin/activate
    python -m pip install -e ".[dev]"
    pytest
    ```

## Add a New Check

1. Add a `check_*` function in `src/mlenvdoctor/diagnose.py`.
2. Return one or more `DiagnosticIssue` values through `make_issue(...)`.
3. Give every issue a stable `check_id`, `category`, `severity`, `fix`, and `evidence`.
4. Add the check to `diagnose_env(...)`.
5. Add or update tests in `tests/`.
6. Document the signal in `docs/checks.md`.

```python
issues.append(
    make_issue(
        name="Example Runtime",
        status="WARN - Runtime fallback detected",
        severity="warning",
        fix="Install the accelerator-enabled runtime",
        check_id="example_runtime",
        category="runtime",
        evidence=["backend=cpu"],
    )
)
```

!!! info "Good checks are specific"
    A useful check says what failed, why it matters, what evidence was observed, and what command verifies the fix.

## Test Before Opening a PR

```bash
python -m black src tests
python -m ruff check src tests
python -m pytest
python -m pip install -r requirements-docs.txt
mkdocs build --strict
```

## Documentation Preview

```bash
python -m pip install -r requirements-docs.txt
mkdocs serve
```

Open `http://127.0.0.1:8000/`.
