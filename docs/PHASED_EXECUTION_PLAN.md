# ML Environment Doctor Phased Execution Plan

This document turns the long-range strategy in [KILLER_PRODUCT_PLAN.md](KILLER_PRODUCT_PLAN.md) into execution phases we can build against incrementally.

## Phase 1 - Brain and UX

Primary goal:

- make `doctor` genuinely useful, less repetitive, and more trustworthy

Scope:

- rule-based root-cause inference above detailed checks
- confidence and linked evidence in doctor findings
- stronger separation between `doctor` and `diagnose`
- more verification-first guidance
- stabilize the detailed diagnostic schema shape

Definition of done:

- `doctor` groups related issues into root-cause findings
- each finding explains likely cause, best next fix, confidence, and verification
- `diagnose` remains the evidence/report command
- tests cover root-cause grouping and summary output

## Phase 2 - Hardware and Framework Breadth

Primary goal:

- remove the current NVIDIA/Linux bias

Scope:

- backend/vendor detection
- CPU-only, MPS, ROCm, DirectML, and WSL2-aware logic
- more useful framework compatibility knowledge
- broader workflow stack profiles

Definition of done:

- users on non-NVIDIA platforms stop receiving misleading CUDA-first guidance
- framework guidance is clearly platform-aware

## Phase 3 - Safe Repair and Docker Validation

Primary goal:

- make `fix` and container workflows trustworthy enough for important machines

Scope:

- formal `fix --plan`, `fix --apply`, and `fix --verify` lifecycle
- risk classification and explicit before/after verification
- partial rollback planning where feasible
- stronger Docker GPU runtime validation and clearer failure messages

Definition of done:

- users can see exactly what a repair action will do
- verification results are explicit and comparable before and after changes

## Phase 4 - Integrations and Ecosystem

Primary goal:

- make `mlenvdoctor` part of the standard AI tooling stack

Scope:

- richer MCP tool family
- reusable CI integration
- stable schemas for assistants and pipelines
- clearer contributor and ecosystem docs

Definition of done:

- the package is easy to use from assistants and CI systems
- bug reports increasingly arrive with structured diagnostics attached

## Current Build Focus

Active build slice:

- Phase 1

Immediate implementation priorities:

1. root-cause grouping for `doctor`
2. confidence and linked-evidence summaries
3. stronger verification-oriented output
4. schema stabilization for actionable findings across CLI, exports, and MCP

The remaining phases should build on a strong local-user product rather than attempt broad ecosystem expansion too early.
