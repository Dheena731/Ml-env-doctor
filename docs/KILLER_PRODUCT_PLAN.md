# ML Environment Doctor Killer Product Plan

## 1. Executive Thesis

`mlenvdoctor` should become the default ML environment triage and recovery tool for AI developers.

That position will not be won by adding more commands or longer reports. It will be won by combining:

- clearer root-cause explanations than generic environment tools
- safer fix flows than ad hoc shell debugging
- broader ML-stack awareness than package-only or framework-only tools
- better local developer UX than CI-only or enterprise-first alternatives

### Strategic Defaults

- Primary user: individual AI developers
- Planning horizon: 6-12 months
- Business posture: open-source-first

### Product Shape

- `doctor` = triage assistant
- `diagnose` = evidence and report engine
- `fix` = safe repair assistant
- `report` = shareability layer
- `mcp` = future integration surface

The product should optimize first for the person who is blocked on a local machine and needs to know what to do next. Team, CI, and assistant integrations matter, but they should grow on top of a strong local product rather than define it too early.

## 2. Users and Jobs To Be Done

The product should be built around reducing “I do not know what to do next” moments.

### 2.1 Individual AI Engineer

Trigger moment:

- local training or inference setup fails
- `torch.cuda.is_available()` is false
- a framework imports but still does not use the expected accelerator
- Docker works but GPU containers do not

Desired outcome:

- identify the likely root cause quickly
- get one best next fix
- verify recovery without guessing

Why current tooling is frustrating:

- framework docs are fragmented
- package managers explain dependency state but not ML runtime failures
- GPU and driver issues require too much manual correlation
- environment tools often report facts without prioritizing action

Success with `mlenvdoctor` looks like:

- a blocked engineer runs `doctor`
- gets a short explanation and a trustworthy next step
- runs a fix or manual command
- verifies recovery without searching multiple docs

### 2.2 New Developer / Learner

Trigger moment:

- first machine setup for ML or LLM development
- virtual environment confusion
- CUDA or framework installation feels intimidating
- environment appears “partly working” and they do not know what matters

Desired outcome:

- understand what is broken in plain language
- avoid unsafe or random installation steps
- get guided recovery

Why current tooling is frustrating:

- too much jargon
- too many install variants
- conflicting internet advice
- environment debugging feels like trial and error

Success with `mlenvdoctor` looks like:

- the tool explains the problem in simple terms
- it suggests a small number of safe actions
- the user can recover without becoming a framework expert

### 2.3 Team Maintainer / CI Owner

Trigger moment:

- teammate machine setup is failing
- CI runner fails intermittently or only on some hosts
- issue reports lack enough environment context

Desired outcome:

- get structured evidence
- reduce support back-and-forth
- standardize team debugging flow

Why current tooling is frustrating:

- issue reports are inconsistent
- shell transcripts are noisy and incomplete
- CI logs rarely include enough environment detail

Success with `mlenvdoctor` looks like:

- structured diagnostic artifacts are attached by default
- `doctor` provides a quick summary for humans
- `diagnose` and `report` provide reproducible evidence for maintainers

## 3. Competitive Landscape

`mlenvdoctor` should not try to beat every competitor on that tool’s home turf. It should win by combining:

- ML-specific diagnosis
- causal explanations
- safe fix verification
- cross-stack coverage

### 3.1 `conda doctor`

What it does well:

- checks package and environment integrity
- useful for Conda-specific environment health

Where it is narrower or stronger:

- stronger for Conda internals
- weaker for GPU, CUDA, framework runtime, and ML-stack diagnosis

What `mlenvdoctor` must do better to win:

- explain ML runtime failures instead of only environment corruption
- connect package state to framework and accelerator behavior

### 3.2 `pip check` and `pip inspect`

What they do well:

- identify Python dependency consistency problems
- provide package-level metadata and conflict visibility

Where they are narrower or stronger:

- stronger at Python package graph inspection
- much weaker for GPU, framework runtime validation, and repair guidance

What `mlenvdoctor` must do better to win:

- connect dependency state to practical ML execution failures
- explain what package or build mismatch matters for the user’s workload

### 3.3 PyTorch `collect_env`

What it does well:

- produces detailed environment reports
- useful for support and bug filing

Where it is narrower or stronger:

- stronger as a PyTorch-specific reporting utility
- weaker at triage, fixing, and multi-framework workflows

What `mlenvdoctor` must do better to win:

- provide prioritization, likely causes, and next actions
- support TensorFlow, JAX, Docker, and higher-level ML workflows

### 3.4 Hugging Face `accelerate env` / `accelerate test`

What they do well:

- validate parts of the Hugging Face and distributed training setup
- useful inside HF-centric workflows

Where they are narrower or stronger:

- stronger for `accelerate`-specific and distributed context
- weaker as a general ML environment doctor across frameworks and platforms

What `mlenvdoctor` must do better to win:

- cover broader environment and runtime readiness
- explain failures outside the HF launcher ecosystem

### 3.5 General Environment Tools such as `envhealth`

What they do well:

- broad system checks
- useful for general machine health baselines

Where they are narrower or stronger:

- stronger at generic system-level inspection
- weaker for ML-specific causal guidance and repair-oriented UX

What `mlenvdoctor` must do better to win:

- understand ML frameworks, accelerator backends, and model workflow constraints
- produce recommendations tied to real ML development paths

### 3.6 Competitive Summary

| Tool | Core strength | Weakness for ML debugging | What `mlenvdoctor` must outperform |
| --- | --- | --- | --- |
| `conda doctor` | Conda environment integrity | Weak on GPU and framework runtime diagnosis | Root-cause ML guidance beyond package corruption |
| `pip check` / `pip inspect` | Python dependency consistency and metadata | No accelerator or runtime reasoning | Practical fix guidance tied to execution failures |
| PyTorch `collect_env` | Detailed PyTorch environment reporting | No triage engine, no fix flow, no multi-framework view | Prioritization, recovery path, broader ML-stack support |
| `accelerate env` / `accelerate test` | HF and distributed setup visibility | Narrow scope outside HF workflows | General-purpose ML environment triage |
| General tools such as `envhealth` | Broad system checks | Weak ML-specific interpretation | ML-native explanations and verification-oriented repair |

## 4. Current-State Gap Analysis

The future-plan ambition is strong, but the current product is only partway there.

### 4.1 Pillar A - Causal Diagnosis Engine

Target state:

- a small set of root causes with confidence and evidence
- cross-check inference, not just isolated failing checks
- prioritized next-step recommendations

Current state:

- `doctor` and `diagnose` are now partially separated
- richer issue fields exist, including likely cause and verification metadata
- fallback logic can summarize issues for `doctor`

Gap:

- no true inference engine yet
- most likely-cause logic is static mapping by check type
- no confidence model based on evidence strength
- little cross-check reasoning between related failures

Product risk if ignored:

- the tool feels repetitive or generic
- experienced users lose trust quickly
- `doctor` remains a nicer wrapper around symptom reporting rather than a true triage assistant

Recommended next move:

- build a rule-based inference layer above raw checks
- group related symptoms into root causes
- score confidence using number and strength of supporting signals

### 4.2 Pillar B - Hardware- and Platform-Agnostic Support

Target state:

- first-class support across NVIDIA, ROCm, DirectML, MPS, and CPU-only environments
- OS-aware guidance for Linux, Windows, WSL2, and macOS

Current state:

- current logic is still strongly NVIDIA/CUDA oriented
- some platform compatibility exists in utilities and CI matrix
- non-NVIDIA backends are not meaningfully diagnosed

Gap:

- no ROCm path
- no DirectML path
- no MPS path
- no WSL2-specific intelligence
- many fix suggestions still assume CUDA/NVIDIA

Product risk if ignored:

- misleading recommendations on non-NVIDIA systems
- the product cannot credibly position itself as universal

Recommended next move:

- add backend and vendor detection as a foundational layer
- branch guidance by hardware backend and OS before expanding fixes

### 4.3 Pillar C - Framework and Tooling Coverage

Target state:

- version-aware compatibility knowledge across PyTorch, TensorFlow, JAX, Keras, and HF tooling
- workflow-aware stacks for common use cases
- more practical support for distributed workflows

Current state:

- PyTorch, TensorFlow/Keras, JAX/Flax, and core HF libraries are already checked
- `stack llm-training` exists

Gap:

- no formal compatibility matrix
- no official driver/toolkit/framework compatibility knowledge layer
- workflow stacks are still limited
- no distributed sanity checks
- current checks are more presence/version oriented than workflow-oriented

Product risk if ignored:

- breadth appears decent, but depth feels shallow
- users still need external docs for real compatibility decisions

Recommended next move:

- introduce compatibility data and stack profiles
- add distributed and HF workflow reasoning after the compatibility layer exists

### 4.4 Pillar D - Safe, Opinionated Repair Engine

Target state:

- explicit plan/apply/verify lifecycle
- risk levels for actions
- before/after verification
- rollback where feasible

Current state:

- `fix` now supports planning, dry-run, apply, and verification
- requirements and Conda generation work
- installation can be attempted and verified

Gap:

- no dedicated `--plan` or `--verify` public subflow
- no action risk classification
- no rollback
- no formal before/after diff presentation
- repair is still mostly package and env generation focused

Product risk if ignored:

- users hesitate to trust automated repair
- the word “doctor” overpromises relative to the repair depth

Recommended next move:

- formalize fix lifecycle commands
- classify actions by risk
- make verification output more explicit and trust-building

### 4.5 Pillar E - Deep Integrations

Target state:

- full MCP surface
- reusable CI integrations
- schemas that AI assistants and tools can depend on
- better ecosystem fit

Current state:

- MCP exists as a stub with a minimal tool surface
- CI exists for the project itself
- exports and reports already provide a useful foundation

Gap:

- no structured MCP tool family for diagnosis, fixing, and reporting
- no official GitHub Action for users
- no examples for broader CI systems
- no IDE integration path yet

Product risk if ignored:

- the product remains a CLI utility rather than part of the AI tooling stack
- team adoption grows more slowly than local experimentation

Recommended next move:

- expand MCP and publish stable schemas
- ship a reusable GitHub Action and example CI recipes

## 5. Product Strategy and Moat

The moat is not:

- more checks
- more commands
- more output formats

The moat is:

- better causal inference
- sharper next-step guidance
- safer verification-driven fixes
- ML-native cross-framework understanding
- usable output for both humans and automation

Every major feature should follow the same principle:

1. explain the likely cause
2. recommend the best next fix
3. prove whether recovery worked

If a feature cannot improve root-cause clarity, repair safety, or trustworthy verification, it should not be prioritized over these fundamentals.

## 6. 6-12 Month Roadmap

### Phase 1 - Brain and UX

Goal:

- make `doctor` genuinely useful and non-repetitive

Must-ship outcomes:

- rule-based causal layer over detailed checks
- confidence scoring for root causes
- `doctor` output limited to prioritized actionable findings
- frozen JSON schema v1 for detailed diagnostics
- before/after verification summaries for fixes

Success metrics:

- fewer generic fix suggestions
- better signal-to-noise in `doctor`
- first-time users can follow the next step without external docs

### Phase 2 - Hardware and Framework Breadth

Goal:

- remove the current NVIDIA/Linux bias

Must-ship outcomes:

- GPU and backend vendor detection
- MPS, ROCm, DirectML, and CPU-only logic paths
- OS-aware recommendations for Linux, Windows, WSL2, and macOS
- framework compatibility knowledge for PyTorch, TensorFlow, and JAX
- more useful Hugging Face and `accelerate` diagnostics

Success metrics:

- users on non-NVIDIA systems no longer receive misleading guidance
- output is clearly platform-aware

### Phase 3 - Safe Repair and Docker Validation

Goal:

- make `fix` trustworthy enough to use on important machines

Must-ship outcomes:

- formal `fix --plan`, `fix --apply`, and `fix --verify`
- explicit risk levels for actions
- before/after verification diff
- partial rollback strategy where feasible
- stronger Docker runtime validation and GPU passthrough diagnosis

Success metrics:

- users feel safe running repair actions
- Docker failures become more explainable

### Phase 4 - Integrations and Ecosystem

Goal:

- turn the CLI into part of the normal AI tooling stack

Must-ship outcomes:

- richer MCP tool surface
- CI examples and an official reusable GitHub Action
- stable schemas for assistant and tooling use
- better docs and contributor pathways

Success metrics:

- project is embedded in assistants, CI, and team workflows
- issue reports arrive with structured diagnostics by default

## 7. Competitive Priorities

If resources are constrained, prioritize in this order:

1. causal diagnosis and root-cause grouping
2. platform universality
3. trustworthy fix lifecycle
4. MCP and CI integration
5. Docker and workflow breadth

This ordering matches an open-source, individual-developer-first strategy. The fastest way to become a standout product is to make the local developer experience substantially better before broadening the ecosystem surface.

## 8. Public Product Contract Changes

The interface direction should be clear even before every feature is implemented.

### Command Roles

- `doctor` remains the top-level triage command
- `diagnose` remains the detailed evidence and export command
- `fix` evolves toward explicit subflows:
  - `--plan`
  - `--apply`
  - `--verify`
  - later `--rollback`
- `report` continues to be the report and artifact command

### MCP Direction

`mcp` should evolve from a stub into a structured tool surface:

- `diagnose_environment`
- `doctor_summary`
- `get_fix_plan`
- `verify_fix`
- `generate_dockerfile`
- `export_report`

### Schema Direction

Every actionable finding should eventually include:

- problem
- likely cause
- best fix
- verify steps
- confidence
- linked evidence

That schema direction supports both human triage and machine integration.

## 9. Metrics and Acceptance Criteria

The project should measure progress across usefulness, adoption, and trust.

### 9.1 Product Usefulness

- reduction in repetitive or generic findings
- more actionable first recommendation success
- fewer steps from diagnosis to recovery

### 9.2 Adoption

- CLI installs
- GitHub stars, issues, and public usage stories
- CI and assistant integrations

### 9.3 Trust

- fewer misleading instructions across platforms
- higher verification success after fixes
- more reproducible issue reports

### Qualitative Acceptance Signals

- AI engineers say outputs save time
- newer users can act without external search
- maintainers receive better bug reports and environment evidence

## 10. Execution Guidance for Maintainers

Use this as the operating model for roadmap decisions:

- do not add new checks unless they improve root-cause clarity or repair safety
- prefer fewer, better recommendations over many generic ones
- avoid claiming support that is not implemented
- keep `doctor` and `diagnose` non-overlapping
- treat every new fix as a trust feature, not just an automation feature

## 11. Relationship to the Existing Future Plan

[mlenvdoctor-future-plan.md](/home/dhinakaran/Ml-env-doctor/mlenvdoctor-future-plan.md) should remain the concise north-star note.

This document is the expanded strategy source of truth that:

- preserves the same high-level phases
- adds competitor comparison
- adds current-state gap analysis
- adds execution priorities
- adds success metrics
- adds product contract decisions

Together, the two documents should serve different roles:

- `mlenvdoctor-future-plan.md` = compact strategic direction
- `docs/KILLER_PRODUCT_PLAN.md` = decision-ready product plan
