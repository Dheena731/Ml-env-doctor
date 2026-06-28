# Usage

mlenvdoctor exposes a CLI for daily use and a small Python API for tools that need structured diagnostic objects.

## CLI Reference

| Command | Use when | Example |
| --- | --- | --- |
| `doctor` | You want the highest-signal summary | `mlenvdoctor doctor --guided` |
| `diagnose` | You need the complete table or exports | `mlenvdoctor diagnose --full --json issues.json` |
| `report` | You want JSON and HTML reports together | `mlenvdoctor report --output-dir mlenvdoctor-report` |
| `fix` | You want a safe plan or generated requirement files | `mlenvdoctor fix --plan` |
| `dockerize` | You need a Dockerfile for a model stack | `mlenvdoctor dockerize tinyllama --service` |
| `stack` | You want recommended dependency pins | `mlenvdoctor stack llm-training` |
| `mcp serve` | You want assistant/tool integration | `mlenvdoctor mcp serve` |

## Diagnose

=== "Windows PowerShell"

    ```powershell
    mlenvdoctor diagnose --full --json issues.json --html issues.html
    ```

=== "macOS / Linux"

    ```bash
    mlenvdoctor diagnose --full --json issues.json --html issues.html
    ```

=== "CI"

    ```bash
    mlenvdoctor doctor --ci
    ```

=== "Automation JSON"

    ```bash
    mlenvdoctor doctor --json
    ```

## Guided Recovery

```bash
mlenvdoctor doctor --guided
```

Use this for beginners, workshop environments, bootcamp labs, and teammate handoffs. It compresses many low-level checks into the top root-cause finding.

## Python API

```python
from mlenvdoctor.api import diagnose

issues = diagnose(full=True)

for issue in issues:
    if issue.status.startswith(("FAIL", "WARN")):
        print(issue.check_id, issue.severity, issue.fix)
```

The API returns `DiagnosticIssue` objects with stable fields:

| Field | Meaning |
| --- | --- |
| `name` | Human-readable finding name |
| `status` | `PASS`, `WARN`, `FAIL`, or `INFO` with detail |
| `severity` | `critical`, `warning`, or `info` |
| `check_id` | Stable machine-readable check identifier |
| `category` | Runtime area such as `pytorch`, `gpu`, `docker`, or `system` |
| `fix` | Recommended action |
| `evidence` | Short evidence lines collected during the check |

!!! info "Use JSON for external automation"
    Use `mlenvdoctor doctor --json` for compact triage. Use `mlenvdoctor diagnose --json -` when another tool needs all raw checks.
