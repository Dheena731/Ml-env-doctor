# Reports and Exports

## Export from `diagnose`

```bash
mlenvdoctor diagnose --json diagnostics.json
mlenvdoctor diagnose --csv diagnostics.csv
mlenvdoctor diagnose --html diagnostics.html
```

## Shareable Bundle from `report`

```bash
mlenvdoctor report
mlenvdoctor report --quick --output-dir artifacts/mlenvdoctor
```

## Machine-Readable JSON

```bash
mlenvdoctor diagnose --json -
```

The JSON includes:

- issues with `check_id`, `category`, `confidence`, `evidence`, and `mismatch_code`
- `doctor_summary`
- `runtime_context`
- `summary` counts
- `exit_code`
- suggested `fixes`

Generated reports are local artifacts. Keep them out of git unless a test fixture explicitly needs one.
