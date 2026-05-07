# Fix workflow

`fix` is intentionally explicit: plan first, apply intentionally, verify after.

## Plan

```bash
mlenvdoctor fix --plan
```

## Dry-run (no changes)

```bash
mlenvdoctor fix --dry-run
```

## Apply

```bash
mlenvdoctor fix --apply
mlenvdoctor fix --apply --yes
```

## Verify

```bash
mlenvdoctor fix --verify
```

## Rollback (file snapshot restore)

If a fix wrote files (for example requirements/environment files) and you want to restore the last snapshot:

```bash
mlenvdoctor fix --rollback
```

