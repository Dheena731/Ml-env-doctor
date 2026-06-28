# Quickstart

This path gives value in under a minute: install, diagnose, export evidence.

## 30-Second Demo

```bash
pip install mlenvdoctor
mlenvdoctor doctor --guided
mlenvdoctor report
```

Expected shape of the output:

```text
Running ML Environment Diagnostics...

Guided Recovery

What failed: PyTorch cannot use the available GPU
Likely cause: NVIDIA is visible, but the installed PyTorch build cannot use CUDA.
Step 1: Reinstall the CUDA-enabled PyTorch wheel for this environment.
Step 2: python -c "import torch; print(torch.cuda.is_available())"
```

!!! info "One command, one next action"
    Use `doctor --guided` when you want the shortest path from confusion to a fix. Use `diagnose` when you need the full evidence table.

## Generate Shareable Evidence

```bash
mlenvdoctor diagnose --json issues.json --html issues.html
```

Attach `issues.json` to a CI artifact, ticket, or team chat. Open `issues.html` when you need a human-readable report.

## Fix Safely

Start with a plan:

```bash
mlenvdoctor fix --plan
```

Apply only after reviewing the generated actions:

```bash
mlenvdoctor fix --apply --yes
mlenvdoctor fix --verify
```

!!! warning "Driver and system changes stay manual"
    mlenvdoctor can generate environment files and dependency plans. It does not silently install GPU drivers or mutate system-level CUDA components.

## CI Smoke Check

```bash
mlenvdoctor doctor --ci
mlenvdoctor doctor --json
```

The command returns a non-zero exit code when critical environment issues are present, so it can fail a workflow before training starts.
