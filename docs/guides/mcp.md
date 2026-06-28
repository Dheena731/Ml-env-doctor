# MCP integration

`mlenvdoctor mcp serve` provides a JSON-lines interface over stdin/stdout.

## Start the server

```bash
mlenvdoctor mcp serve
```

## Discover tools

```json
{"tool":"list_tools","arguments":{}}
```

```json
{"tool":"tool_schema","arguments":{"name":"export_report"}}
```

## v1 alias tools

These are supported as stable aliases:

- `diagnose_environment` (alias for `diagnose`)
- `get_fix_plan` (alias for `get_fixes`)
- `verify_fix` (alias for `doctor_summary`)
- `export_report` (alias for `report_bundle`)

## Recommended payload for assistants

Use `doctor_summary` when you need the top triage result, runtime context, exit code, top fix, and verification step.

Use `export_report` or `report_bundle` when you need:

- summary counts
- runtime context
- doctor summary
- fixes
- exit code
