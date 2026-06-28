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

## `doctor --json` shape

When you run:

```bash
mlenvdoctor doctor --json
```

You receive a compact JSON object for triage automation:

- `schema_version`: doctor summary schema version
- `doctor_summary`: prioritized root-cause findings
- `runtime_context`: platform/backend summary
- `summary`: counts (passed/warnings/critical/total)
- `exit_code`: stable exit code (0/1/2)
- `top_fix`: best next action from the highest-priority finding
- `next_verify_step`: first verification command for the top finding
- `metadata`: tool version, schema version, timestamp

