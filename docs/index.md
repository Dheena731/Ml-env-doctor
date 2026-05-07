# ML Environment Doctor

Diagnose machine learning environment issues, export shareable reports, and generate safe starting fixes.

## Who this is for

- **Beginners**: You want clear next steps without digging through CUDA forums.
- **AI engineers**: You want fast, evidence-backed triage and machine-readable output for CI and automation.

## Quick start

```bash
pip install mlenvdoctor

mlenvdoctor doctor --guided
mlenvdoctor diagnose --json -
mlenvdoctor fix --dry-run
```

## Common workflows

### I just want to know what to do next

```bash
mlenvdoctor doctor
```

or beginner-friendly:

```bash
mlenvdoctor doctor --guided
```

### I need evidence and a report to share

```bash
mlenvdoctor diagnose --json diagnostics.json --html diagnostics.html
```

### I want safe starting fixes (plan first)

```bash
mlenvdoctor fix --plan
mlenvdoctor fix --apply --yes
mlenvdoctor fix --verify
```

### I need to integrate with assistants/tools (MCP)

```bash
mlenvdoctor mcp serve
```

See [Guides > MCP integration](guides/mcp.md).

