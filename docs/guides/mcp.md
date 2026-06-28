# MCP Integration

`mlenvdoctor mcp serve` provides a JSON-lines interface over stdin/stdout.

## Start the Server

```bash
mlenvdoctor mcp serve
```

## Discover Tools

```json
{"tool":"list_tools","arguments":{}}
```

```json
{"tool":"tool_schema","arguments":{"name":"export_report"}}
```

## v1 Alias Tools

These are supported as stable aliases:

- `diagnose_environment` (alias for `diagnose`)
- `get_fix_plan` (alias for `get_fixes`)
- `verify_fix` (alias for `doctor_summary`)
- `export_report` (alias for `report_bundle`)

## Recommended Payload for Assistants

Use `export_report` or `report_bundle` to get:

- summary counts
- runtime context
- doctor summary
- fixes
- exit code
