# Next Version Plan: 0.2.0

Version 0.2.0 should turn `mlenvdoctor` from a diagnostic CLI into an automation-ready triage assistant for local developers, CI jobs, and support workflows.

## Release Theme

**Automation-ready triage.** The next version should make the highest-signal result easy to consume from humans, scripts, CI, MCP clients, and future UI surfaces.

## Milestones

1. Machine-readable doctor summary
   - Add `mlenvdoctor doctor --json`.
   - Share the same summary payload between CLI exports and MCP.
   - Include exit code, runtime context, summary counts, top fix, and first verification step.

2. Readiness profiles
   - Add workflow profiles such as `minimal`, `llm-training`, `tensorflow`, `jax`, and `docker-gpu`.
   - Let profiles tune which warnings matter most without hiding raw evidence.
   - Surface the selected profile in JSON exports.

3. Snapshot comparison
   - Add a command that compares two JSON reports and shows what improved, regressed, or stayed blocked.
   - Make the comparison output useful for CI comments and support tickets.

4. Safer fix execution
   - Expand fix planning with clearer preconditions, rollback notes, and verification gates.
   - Keep system-level driver and CUDA mutations manual.

5. Documentation and release polish
   - Document the doctor JSON schema and common CI/MCP patterns.
   - Add migration notes for users moving from 0.1.x to 0.2.0.

## First Slice Started

This branch starts milestone 1 with:

- a reusable doctor-summary payload builder
- `mlenvdoctor doctor --json`
- MCP `doctor_summary` alignment with the same compact shape
- tests and docs for the new automation path

## Acceptance Checks

- `mlenvdoctor doctor --json` emits valid JSON with no Rich formatting.
- Existing `diagnose --json -` payload remains backward compatible.
- MCP `doctor_summary` remains additive and keeps existing fields.
- `python -m pytest` passes.
- `mkdocs build --strict` passes.
