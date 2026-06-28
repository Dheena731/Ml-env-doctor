# Diagnose and Doctor

## `doctor` (triage)

Use when you need a fast answer to:

- What failed?
- Why it likely failed?
- What is the best next fix?
- How do I verify?

Examples:

```bash
mlenvdoctor doctor
mlenvdoctor doctor --ci
mlenvdoctor doctor --guided
mlenvdoctor doctor --json
```

Use `doctor --json` when automation needs only the prioritized triage summary, top fix, and first verification step.

## `diagnose` (evidence)

Use when you need full evidence and exports:

```bash
mlenvdoctor diagnose
mlenvdoctor diagnose --full
mlenvdoctor diagnose --json -
mlenvdoctor diagnose --json diagnostics.json --csv diagnostics.csv --html diagnostics.html
```

## Exit codes

- `0`: no warnings or critical issues
- `1`: warnings present
- `2`: critical issues present

