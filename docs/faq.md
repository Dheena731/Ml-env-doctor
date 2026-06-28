# FAQ

## Does this replace framework install guides?

No. `mlenvdoctor` aims to detect and explain common failures, suggest safe starting fixes, and show how to verify. It does not replace official CUDA/cuDNN/framework documentation.

## Is it safe to run `fix --apply`?

It is designed to be explicit and verifiable. Start with:

```bash
mlenvdoctor fix --plan
mlenvdoctor fix --dry-run
```

If you apply file-generating fixes and want to undo them, try:

```bash
mlenvdoctor fix --rollback
```

## How do I use it in CI?

Use:

```bash
mlenvdoctor doctor --ci
mlenvdoctor diagnose --json diagnostics.json
```

See the workflow template in `.github/workflows/mlenvdoctor-user-ci-template.yml`.
