# Getting Started

## Install

```bash
pip install mlenvdoctor
```

## First run (recommended)

If you're new to ML environment setup, start with guided mode:

```bash
mlenvdoctor doctor --guided
```

## When to use which command

- **`doctor`**: quick triage, best next fix, and how to verify.
- **`diagnose`**: full evidence and exportable report data.
- **`fix`**: explicit plan/apply/verify workflow for safe starting remediation.
- **`report`**: write shareable JSON+HTML artifacts to a folder.
- **`mcp`**: integration surface for assistants and automation.

## Safe defaults

- Prefer `mlenvdoctor fix --plan` before `--apply`.
- Verify changes with `mlenvdoctor fix --verify`.
- If you used file-generating fixes and want to undo them, try `mlenvdoctor fix --rollback`.

