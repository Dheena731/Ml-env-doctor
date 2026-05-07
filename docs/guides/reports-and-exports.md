# Reports and exports

## Export from `diagnose`

```bash
mlenvdoctor diagnose --json diagnostics.json
mlenvdoctor diagnose --csv diagnostics.csv
mlenvdoctor diagnose --html diagnostics.html
```

## Shareable bundle from `report`

```bash
mlenvdoctor report
mlenvdoctor report --quick --output-dir artifacts/mlenvdoctor
```

## Machine-readable JSON

```bash
mlenvdoctor diagnose --json -
```

The JSON includes:\n\n- issues (with `check_id`, `category`, `confidence`, `evidence`, and `mismatch_code`)\n- `doctor_summary`\n- `runtime_context`\n- `summary` counts\n- `exit_code`\n- suggested `fixes`\n+
