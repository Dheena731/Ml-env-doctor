# ML Environment Doctor Improvements Roadmap

## Current Pass

Implemented in this round:

- Machine-readable diagnostic exit codes
  - `0` = healthy
  - `1` = warnings present
  - `2` = critical issues present
- `mlenvdoctor diagnose --json -` for stdout JSON output
- `mlenvdoctor report` to generate shareable HTML and JSON reports
- Copy-paste fix command extraction via shared diagnostics data
- MCP server stub with:
  - `mlenvdoctor mcp serve`
  - tools: `diagnose`, `get_fixes`

Files touched:

- `src/mlenvdoctor/cli.py`
- `src/mlenvdoctor/diagnose.py`
- `src/mlenvdoctor/export.py`
- `src/mlenvdoctor/mcp.py`
- `tests/test_diagnose.py`
- `tests/test_utils.py`

## v0.2.0: Shareability

Goal: make output easy to use in CI and easy to share with teammates.

Status:

- `--json` machine-readable output: done
- machine-readable exit codes: done
- `mlenvdoctor report`: done
- copy-paste fix commands: done
- MCP server stub: done

Follow-up polish:

- add `--json` examples to README
- allow `report --prefix` for deterministic artifact naming in CI
- add a compact `--ci` text mode for log-friendly output

## v0.3.0: Coverage

Goal: broaden ML stack support beyond PyTorch-only workflows.

Must-ship:

- TensorFlow detection
- Keras 3 compatibility checks
- JAX/Flax detection
- TPU/GPU detection for JAX
- OS-specific CUDA path checks for Windows, WSL, and Linux
- `mlenvdoctor stack llm-training`

Suggested implementation order:

1. Add `tensorflow.py` diagnostics helpers for import/version/device checks.
2. Add `jax.py` diagnostics helpers for backend, accelerator, and TPU visibility.
3. Add `platform_checks.py` for:
   - Windows CUDA toolkit paths
   - WSL GPU pass-through heuristics
   - Linux driver/toolkit path checks
4. Add `stack` command to emit recommended dependency sets.
5. Expand export payload schema with framework sections.

Success metric:

- PyTorch + TensorFlow + JAX users all get actionable diagnostics.

## v0.4.0: Adoption

Goal: make the project easier to install, contribute to, and run in automation.

Must-ship:

- Full MCP server:
  - `diagnose_environment`
  - `generate_dockerfile`
  - `test_model`
  - `export_report`
- README overhaul
- GitHub Actions test matrix
- Auto-PyPI publish workflow
- `mlenvdoctor doctor --ci`

Suggested implementation order:

1. Expand `mcp.py` into a proper command router with structured schemas.
2. Add CI-only output mode with compact summaries and stable JSON.
3. Add GitHub Actions for:
   - lint
   - tests on Windows/Linux
   - package build
4. Add publish workflow for tagged releases.
5. Refresh README with troubleshooting snippets and CI examples.

Success metric:

- contributors can run tests locally and in CI with predictable results.

## v1.0.0: Stability

Goal: reach production-trustworthy quality.

Must-ship:

- 90% test coverage
- Windows CI
- security review for generated fix commands
- MCP sandboxing
- docs site
- changelog discipline
- `SECURITY.md`
- `CODEOWNERS`

Suggested implementation order:

1. lock down fix command generation behind validated templates only
2. add integration tests for CLI and report generation
3. publish docs site with:
   - troubleshooting
   - command reference
   - MCP reference
4. add release and security governance files

Success metric:

- users can rely on generated advice in CI and production troubleshooting flows.

## Immediate Next Tasks

Recommended next PRs:

1. Add TensorFlow + Keras 3 diagnostics module.
2. Add JAX/Flax + accelerator detection.
3. Add `stack llm-training`.
4. Add `doctor --ci`.
5. Expand README with `diagnose --json -` and `report` examples.
