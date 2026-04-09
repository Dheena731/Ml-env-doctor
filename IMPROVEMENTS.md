# ML Environment Doctor Improvement Themes

This document tracks the next set of improvements after the current repository cleanup. It reflects the project as it exists now, not historical gaps that are already closed.

## Current Strengths

- Structured diagnostics across the main Python ML frameworks
- Machine-readable report exports with stable exit codes
- Safe fix workflow with dry-run and apply modes
- Configurable Dockerfile generation
- CLI tests that no longer depend on the `mlenvdoctor` executable being pre-installed

## Highest-Value Next Steps

### 1. Deeper Framework Compatibility Matrices

- map specific CUDA, cuDNN, and framework version combinations
- add clearer mismatch detection for Windows, WSL, and Linux
- record framework-native environment details in export metadata

### 2. More Actionable Fix Automation

- add rollback-aware installs for partially applied fixes
- add lockfile generation support
- add stack-specific repair actions instead of mostly requirements generation

### 3. Broader Docker Coverage

- move model profiles and stacks into external config files
- add compose generation and inference-specific profiles
- add validation tests for generated Dockerfiles

### 4. Better Platform Coverage

- expand Windows-specific checks
- add WSL GPU pass-through diagnostics
- improve cloud runner heuristics for common CI platforms

### 5. Integration Depth

- expand the MCP server beyond the current stub
- add optional config-driven defaults for more CLI commands
- add richer report presentation for historical comparisons
