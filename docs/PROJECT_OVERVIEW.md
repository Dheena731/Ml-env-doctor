# ML Environment Doctor Project Overview

## 1. What This Project Is

ML Environment Doctor is a Python CLI package designed to help people understand whether a machine is ready for machine learning work, especially GPU-backed development and fine-tuning workflows.

The project sits in the space between:

- manual environment debugging
- framework-specific setup guides
- CI health checks
- lightweight automation for environment repair

Its core purpose is not to train models itself. Its purpose is to reduce the time spent asking:

- Why is CUDA not visible?
- Why does PyTorch import but still fail?
- Why does this machine work for one framework but not another?
- Why does this CI runner keep failing?
- What is the next best fix to try?

At its best, the package acts like a practical environment triage assistant.

## 2. Why The Project Exists

ML environment setup is frustrating because the failure surface is spread across several layers:

- Python version and virtual environment state
- package installation state
- CUDA driver visibility
- framework build compatibility
- GPU memory availability
- Docker GPU passthrough
- internet access to model registries

Developers often end up jumping between:

- framework docs
- random forum posts
- shell commands
- package reinstall attempts
- container experiments

This project tries to centralize that experience into one CLI.

## 3. Main User Groups

The project is useful for more than one kind of user, and each group wants something slightly different from it.

### 3.1 AI Engineers

An AI engineer usually wants:

- quick confirmation that a machine is usable for training or inference
- a fast way to see whether CUDA, PyTorch, TensorFlow, or JAX are actually working
- root-cause-oriented guidance instead of generic environment advice
- CI-friendly output for runners or remote machines
- a report they can share with teammates

What they care about most:

- speed
- accuracy
- clear next fix
- verification after the fix

### 3.2 New Developers

A new developer usually wants:

- to know whether their machine is set up correctly
- a clear explanation of what is broken
- fewer intimidating commands
- safer defaults
- help without needing deep CUDA knowledge

What they care about most:

- clarity
- reduced confusion
- simple next step
- confidence that they are not making things worse

### 3.3 Maintainers / Team Leads

A maintainer or team lead usually wants:

- a repeatable way to diagnose teammate or CI failures
- structured output for automation
- shared reports
- lower onboarding friction
- fewer environment-specific support messages

What they care about most:

- repeatability
- machine-readable output
- cross-platform reliability
- consistent team workflows

## 4. Current Product Position

Right now the project is a real and useful CLI, but it is still closer to:

- a structured environment inspector

than to:

- a complete repair platform

That is important to say clearly.

Today it already does meaningful work:

- detects several environment problems
- exports reports
- generates requirements and Conda files
- offers a basic fix flow
- scaffolds Dockerfiles
- provides CI-usable status codes

But it still has room to grow before it feels like a deeply intelligent “doctor.”

## 5. Current Product Philosophy

The project is now moving toward a clearer split:

- `doctor`: triage and next-step guidance
- `diagnose`: evidence and detailed inspection

This is a strong direction because it reduces overlap.

### 5.1 `doctor`

`doctor` should answer:

- What failed?
- What is the most likely cause?
- What is the best next fix?
- How do I verify the fix?

`doctor` is the action-oriented command.

### 5.2 `diagnose`

`diagnose` should answer:

- What did the tool observe?
- Which checks passed or failed?
- What evidence supports those results?
- What can I export or share?

`diagnose` is the evidence-oriented command.

## 6. Repository Structure

The repository is intentionally small and centered around the CLI.

```text
Ml-env-doctor/
├── .github/workflows/ci.yml
├── README.md
├── CHANGELOG.md
├── CONTRIBUTING.md
├── IMPROVEMENTS.md
├── IMPROVEMENTS_ROADMAP.md
├── docker/
│   └── README.md
├── docs/
│   └── PROJECT_OVERVIEW.md
├── src/mlenvdoctor/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── constants.py
│   ├── diagnose.py
│   ├── dockerize.py
│   ├── exceptions.py
│   ├── export.py
│   ├── fix.py
│   ├── gpu.py
│   ├── icons.py
│   ├── logger.py
│   ├── mcp.py
│   ├── parallel.py
│   ├── retry.py
│   ├── utils.py
│   └── validators.py
└── tests/
```

## 7. Core Modules Explained

### 7.1 `cli.py`

This is the entrypoint of the package.

It defines the public CLI commands:

- `diagnose`
- `doctor`
- `report`
- `fix`
- `stack llm-training`
- `dockerize`
- `test-model`
- `smoke-test`
- `mcp serve`

This file is responsible for:

- command definitions
- CLI help text
- top-level output behavior
- choosing which internal functions are called

It is the place where product boundaries become visible to the user.

### 7.2 `diagnose.py`

This is the central diagnostics engine.

It contains:

- the `DiagnosticIssue` data model
- the `DoctorFinding` summary model
- individual checks
- grouping and orchestration of checks
- doctor-oriented summarization
- detailed table rendering

This is the most important file in the project because it defines what the package actually knows how to inspect.

### 7.3 `export.py`

This module converts diagnostic issues into:

- JSON
- CSV
- HTML

It also computes:

- summary counts
- stable exit codes

This module makes the package useful for:

- CI
- dashboards
- artifact sharing
- machine parsing

### 7.4 `fix.py`

This module handles the repair-oriented side of the package.

It currently supports:

- fix planning
- dry-run mode
- requirements file generation
- Conda environment file generation
- optional virtual environment creation
- optional requirement installation
- post-fix verification

It is useful, but it is still not a full repair engine yet.

### 7.5 `dockerize.py`

This module generates starter Dockerfiles and service templates.

It supports:

- model-aware Dockerfile generation
- training or service profile generation
- base image override
- Python version override
- stack-based dependency injection

This is valuable as scaffolding, not yet as a full container platform.

### 7.6 `gpu.py`

This module contains runtime-heavy checks and smoke tests, including:

- benchmark operations
- model loading checks
- LoRA smoke test logic

This file moves the package a bit closer to “real runtime validation,” which is useful because imports alone are often misleading.

### 7.7 `mcp.py`

This module exposes a small JSON-lines MCP-style stub.

It currently supports only:

- `diagnose`
- `get_fixes`

This is an early integration surface, not a complete platform integration layer.

### 7.8 Supporting Modules

Other files support the main flow:

- `config.py`: config loading
- `validators.py`: command and input validation
- `utils.py`: shell/process helpers and console helpers
- `logger.py`: logging setup
- `parallel.py`: parallel execution helpers
- `retry.py`: retry logic for transient failures
- `constants.py`: shared values
- `exceptions.py`: custom exception classes

## 8. Current Public Commands

### 8.1 `mlenvdoctor diagnose`

Purpose:

- detailed environment evidence

What it does:

- runs the core checks
- optionally runs extended checks with `--full`
- can export JSON, CSV, or HTML
- can print machine-readable JSON to stdout

When to use it:

- when debugging deeply
- when filing a bug
- when sharing environment state with a teammate
- when collecting evidence in CI

### 8.2 `mlenvdoctor doctor`

Purpose:

- quick triage with next-step guidance

What it does:

- runs diagnostics
- summarizes actionable failures
- explains likely cause
- recommends the next fix
- shows verification steps

When to use it:

- when a person wants to know what to do next
- when the terminal output should be compact
- when CI needs a summary instead of a full report

### 8.3 `mlenvdoctor report`

Purpose:

- produce shareable report artifacts

What it does:

- runs diagnostics
- writes JSON and HTML reports to a folder

### 8.4 `mlenvdoctor fix`

Purpose:

- apply safe starting repair actions

What it does today:

- plans actions
- supports `--dry-run`
- supports `--apply`
- supports `--yes`
- can create a venv
- can generate requirements or Conda files
- can install packages
- can verify after apply

### 8.5 `mlenvdoctor stack llm-training`

Purpose:

- emit a recommended package stack for training workflows

### 8.6 `mlenvdoctor dockerize`

Purpose:

- generate a starter Dockerfile or service scaffold

### 8.7 `mlenvdoctor test-model`

Purpose:

- load a real model and test basic runtime readiness

### 8.8 `mlenvdoctor smoke-test`

Purpose:

- perform a minimal LoRA-style smoke test

### 8.9 `mlenvdoctor mcp serve`

Purpose:

- expose minimal tool access over a JSON-lines interface

## 9. Diagnostics Currently Supported

The current checks cover:

- Python runtime compatibility
- NVIDIA driver visibility
- PyTorch installation
- PyTorch version quality
- PyTorch CUDA visibility
- PyTorch CUDA execution
- common training libraries
- TensorFlow runtime visibility
- TensorFlow execution
- Keras version/readiness
- JAX runtime/backend visibility
- JAX execution
- Flax presence

Extended checks cover:

- GPU memory
- disk space
- Docker GPU support
- internet connectivity to Hugging Face

## 10. Current Data Model

Each detailed issue can now contain:

- `name`
- `status`
- `severity`
- `fix`
- `details`
- `check_id`
- `category`
- `recommendation`
- `likely_cause`
- `verify_steps`
- `confidence`
- `evidence`
- `metadata`

This is important because it lets the project support both:

- human-readable triage
- machine-readable exports

without needing two completely separate diagnostic engines.

## 11. Current Flow Inside The Tool

The high-level flow looks like this:

1. CLI command is invoked
2. `cli.py` selects the correct command behavior
3. `diagnose.py` runs checks and creates `DiagnosticIssue` objects
4. output is either:
   - summarized for `doctor`
   - rendered in detail for `diagnose`
   - exported through `export.py`
   - passed into fix planning
5. optional fix/report/docker actions are performed

## 12. What The Project Already Does Well

The project already has a few strong qualities:

- clear CLI packaging
- small and understandable codebase
- structured diagnostic model
- support for multiple ML frameworks
- stable exit codes
- report exports
- basic automation paths
- CI support

These are good foundations.

## 13. Where The Project Still Feels Weak

This is the honest part.

The project still feels weaker than its ambition in a few areas:

### 13.1 Root Cause Depth

Many checks still identify symptoms better than they identify exact root cause.

Example:

- “CUDA not available” is useful
- “CUDA not available because your PyTorch build is CPU-only while the driver is visible” is much better

### 13.2 Repetition Risk

Without careful product boundaries, commands can start to feel like repeated wrappers around the same checks.

That is why the `doctor` vs `diagnose` split matters a lot.

### 13.3 Fix Quality

The package has a fix flow, but it is still early compared to what users may expect from the word “doctor.”

Users often expect:

- precise repair
- automatic recovery
- rollback safety
- reliable verification

The project is not fully there yet.

### 13.4 Docker Maturity

Docker support is useful but still template-oriented.

It helps with:

- starting points

more than:

- production container lifecycle management

### 13.5 Platform Specificity

Different ML environments break differently on:

- Linux
- Windows
- WSL
- remote servers
- CI runners

The project will need more platform-specific intelligence to feel truly sharp.

## 14. What AI Engineers Will Expect From This Tool

A strong AI engineer will expect this project to do the following well:

- detect framework and CUDA mismatches accurately
- identify the likely root cause quickly
- recommend the highest-value next command
- verify whether a fix actually changed the environment
- avoid noisy or generic advice
- produce machine-readable data for automation

If the tool repeats generic install commands without explaining why, experienced users will lose trust quickly.

## 15. What New Developers Will Expect From This Tool

Newer users will expect:

- simpler language
- less jargon
- guided recovery
- fewer choices at each step
- confidence that the tool is not making unsafe changes

If the tool overwhelms them with raw check output, they may still not know what to do next even if the diagnosis is technically correct.

## 16. What Makes This Project Valuable Long Term

This project becomes truly valuable if it evolves from:

- a list of checks

into:

- a reliable environment decision and repair assistant

The long-term value is not just:

- “we detect a lot of things”

It is:

- “we help people recover faster from ML environment failures”

That is a much stronger product goal.

## 17. Strategic Directions For Improvement

These are the most meaningful directions for the project from here.

### 17.1 Make `doctor` More Opinionated

`doctor` should not behave like a mini report.

It should:

- prioritize
- simplify
- recommend one best next step
- verify that the fix worked

### 17.2 Make Checks More Causal

Each check should ideally produce:

- what failed
- why it likely failed
- what to do next
- how to verify

This is more useful than generic fix strings.

### 17.3 Add Cross-Check Intelligence

Many root causes only become obvious by combining observations.

Examples:

- driver visible + PyTorch CPU-only build
- TensorFlow installed + zero GPU devices + working NVIDIA driver
- Docker installed + daemon works + container GPU fails

The package should infer these combined explanations instead of presenting isolated facts only.

### 17.4 Improve New User Guidance

New developers benefit from:

- clearer wording
- safer suggestions
- guided fix sequences
- fewer overwhelming outputs

### 17.5 Tighten The Automation Story

If `fix` becomes more powerful, it should also become more trustworthy:

- dry-run first
- explicit action plan
- apply safely
- verify
- rollback where possible

### 17.6 Deepen CI and Team Use Cases

Teams will value:

- stable JSON schema
- deterministic reports
- compact CI summaries
- easier support workflows

## 18. Recommended Product Direction

The strongest direction for the project is:

### `doctor` = triage assistant

- compact
- opinionated
- actionable
- user-focused

### `diagnose` = evidence engine

- detailed
- exportable
- support-friendly
- machine-readable

### `fix` = safe repair assistant

- planned
- explicit
- verifiable

### `report` = shareability layer

- portable
- structured
- useful in CI or team debugging

## 19. Current Development Priorities I Would Recommend

If the goal is to make this project genuinely useful to AI engineers and new developers, the best next priorities are:

1. Improve root-cause inference quality
2. Make `doctor` outputs more trusted and less repetitive
3. Increase verification quality after fixes
4. Add cross-platform environment intelligence
5. Improve onboarding and support documentation
6. Make Docker support broader and more data-driven

## 20. Final Summary

ML Environment Doctor is a promising CLI project with a good structural foundation.

Its strengths are:

- a clear code layout
- a workable command surface
- multiple framework checks
- export/report support
- a growing separation between triage and evidence

Its current weakness is not that it does nothing useful.

Its weakness is that it still needs to become:

- more causal
- more opinionated
- less repetitive
- more trustworthy in recovery guidance

If the project continues moving toward:

- “explain the likely cause, recommend the best next fix, and show how to verify it”

then it can become genuinely valuable for both experienced AI engineers and newer developers.

## 21. Related Strategy Documents

For the long-range product strategy and competitive roadmap, see:

- [KILLER_PRODUCT_PLAN.md](/home/dhinakaran/Ml-env-doctor/docs/KILLER_PRODUCT_PLAN.md)
- [mlenvdoctor-future-plan.md](/home/dhinakaran/Ml-env-doctor/mlenvdoctor-future-plan.md)
