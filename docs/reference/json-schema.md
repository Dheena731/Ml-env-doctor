# JSON schema (practical)

When you run:

```bash
mlenvdoctor diagnose --json -
```

You receive a single JSON object with these top-level keys:

- `issues`: list of detailed checks
- `doctor_summary`: grouped findings intended for triage
- `runtime_context`: platform/backend summary
- `summary`: counts (passed/warnings/critical/total)
- `exit_code`: stable exit code (0/1/2)
- `fixes`: suggested fix commands
- `metadata` (when enabled): tool version, schema version, timestamp

## `issues[]` shape

Each issue includes:

- `name`, `status`, `severity`
- `check_id`, `category`
- `fix`, `recommendation`
- `likely_cause`, `verify_steps`
- `confidence`, `evidence`
- `mismatch_code` (when applicable)
- `metadata`
