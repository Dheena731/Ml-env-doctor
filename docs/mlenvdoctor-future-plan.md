# mlenvdoctor Future-Proof Product Plan

## 1. Product North Star

**Positioning**  
"mlenvdoctor is the universal ML environment health system: it explains failures, applies safe fixes, and verifies recovery across GPUs/CPUs, frameworks, Docker, cloud, and CI — via CLI and MCP."

Every feature decision should answer:
- Does this improve **root-cause clarity**?
- Does this make **fixing safer and faster**?
- Does this work across **more platforms and stacks than other env doctor tools**?

---

## 2. Core Feature Pillars

### Pillar A – Causal Diagnosis Engine

**Goal**: Move from a list of failing checks to a small set of **root causes** with confidence and evidence.

Each root cause should include:
- What failed (short label)
- Why it likely failed (causal explanation)
- What to do next (concrete commands)
- How to verify the fix
- Confidence level
- Linked evidence from checks

**Key work items**:
- Build a **rule/inference layer** on top of existing checks:
  - Example rules:
    - Driver visible + `torch` installed + `torch.version.cuda is None` ⇒ CPU-only PyTorch wheel on GPU-capable system ⇒ suggest reinstalling correct CUDA wheel.
    - TensorFlow installed + `nvidia-smi` OK + `tf.config.list_physical_devices('GPU') == []` ⇒ TensorFlow installed without GPU support ⇒ suggest `tensorflow[and-cuda]` path on Linux.
    - Docker + NVIDIA toolkit present + `docker run --gpus all` fails ⇒ container runtime misconfigured ⇒ point to NVIDIA Container Toolkit + daemon restart steps.
  - Cross-checks combining GPU memory, VRAM, model size, and config to detect oversubscription.
- Implement a **confidence score** per root cause based on number/strength of supporting checks.

**Outcome**: Feels like an actual "doctor" explaining likely causes, not just a checker printing symptoms.

---

### Pillar B – Hardware- and Platform-Agnostic Support

**Goal**: First-class support across:
- Hardware: NVIDIA (CUDA), AMD (ROCm), Intel/DirectML, Apple Silicon (MPS), CPU-only.
- OS: Linux, Windows, WSL2, macOS.

**Key work items**:
- Hardware detection:
  - Detect GPU vendor (NVIDIA/AMD/Intel/Apple) and available backends.
- Backend-specific checks:
  - CUDA: toolkit, driver, GPU visibility.
  - ROCm: ROCm runtime presence, AMD GPU visibility.
  - DirectML: PyTorch/TF DirectML validation on Windows.
  - MPS: `torch.backends.mps.is_available()` and basic sanity test on Apple.
- OS-specific guidance:
  - Linux: CUDA toolkit and NVIDIA Container Toolkit install/validation; TF/JAX GPU installs.
  - Windows: clear guidance between WSL2 + CUDA path vs native DirectML; explain TensorFlow GPU limitations.
  - macOS: steer towards MPS + CPU-friendly workflows (no fake CUDA instructions).

**Outcome**: Becomes the only truly **universal** ML environment doctor, not tied to NVIDIA-only flows.

---

### Pillar C – Framework & Tooling Coverage

**Goal**: Cover modern ML/LLM workflows beyond PyTorch.

**Key work items**:
- Maintain a versioned **compatibility matrix**:
  - PyTorch ↔ CUDA toolkit ↔ minimum driver.
  - TensorFlow ↔ CUDA/cuDNN combos (per official docs).
  - JAX ↔ CUDA/cuDNN combos on Linux.
- Extend checks for:
  - TensorFlow, JAX, Keras 3.
  - Hugging Face stack: `transformers`, `accelerate`, `peft`, `trl`, `datasets`, `bitsandbytes`.
- Add workflow profiles:
  - `stack llm-training`
  - `stack inference`
  - `stack cpu-only`
  - `stack research`
- Basic distributed sanity checks:
  - `accelerate config` issues, mismatched world size, etc.

**Outcome**: Clearly optimized for real LLM + HF workflows, not just generic PyTorch/TF imports.

---

### Pillar D – Safe, Opinionated Repair Engine

**Goal**: Make `fix` feel like a calm, trustworthy SRE: always shows plan, applies safe actions, and proves success.

**Key work items**:
- Fix lifecycle:
  - `fix --plan` → show explicit planned actions grouped by risk (low/medium/high).
  - `fix --apply` → execute low/medium-risk actions with clear prompts.
  - `fix --verify` → re-run key checks and show before/after diff.
  - `fix --rollback` → where possible, restore previous environment snapshot.
- Risk classification:
  - Low risk: new venv, adding missing libraries, generating env files.
  - Medium: upgrading packages, changing minor versions.
  - High: driver/toolkit changes → provide script + docs, but do not auto-run.
- Use **official install guidance**:
  - TF GPU on Linux: `tensorflow[and-cuda]` instead of ad-hoc instructions.
  - JAX GPU: wheels / extra index as per official docs.
  - NVIDIA Container Toolkit: align with current NVIDIA install docs.

**Outcome**: Users are comfortable running `fix` on important machines and in CI.

---

### Pillar E – Deep Integrations (MCP, CI, IDE)

**Goal**: Make mlenvdoctor a default piece of tooling in:
- AI assistants (via MCP),
- CI/CD pipelines,
- Editor/IDE environments.

**Key work items**:
- **MCP server**:
  - Tools exposed: `diagnose_environment`, `doctor_summary`, `get_fix_plan`, `verify_fix`, `generate_dockerfile`, `export_report`.
  - Provide clean JSON schemas and narrow, well-named tools.
- **CI/CD integration**:
  - Official GitHub Action (and docs) to run diagnosis + artifact export.
  - Example pipelines for GitHub, GitLab, Azure.
- **IDE integration (later phase)**:
  - VS Code extension that runs `doctor` and renders results in a panel.

**Outcome**: Becomes part of the standard stack, not just an optional CLI.

---

## 3. Phased Execution Plan

### Phase 1 – Brain & UX (4–6 weeks)

**Focus**: Causal engine + `doctor` redesign.

- Implement causal rule engine above existing checks.
- Redesign `doctor` output to show:
  - Top N root causes with explanations and commands.
  - One "do this next" command.
- Improve `fix` to re-run verification checks and show before/after snapshot.
- Freeze JSON schema v1 for `diagnose` and `report`.

**Success criteria**:
- Expert users say outputs are accurate and non-noisy.
- Beginners can follow `doctor` without needing external searches.

---

### Phase 2 – Hardware & Framework Breadth (4–6 weeks)

**Focus**: Universal hardware/platform and multi-framework coverage.

- Implement GPU vendor and backend detection.
- Add OS-specific logic paths (Linux, Windows, WSL2, macOS).
- Add TF/JAX compatibility knowledge and checks.
- Enrich HF/Accelerate stack diagnostics.

**Success criteria**:
- Handles NVIDIA, AMD, DirectML, MPS, and CPU-only scenarios gracefully.
- Clear, correct guidance across Linux/Windows/macOS.

---

### Phase 3 – Safe Repair & Docker Validation (3–4 weeks)

**Focus**: Trustworthy `fix` and strong container validation.

- Implement `fix --plan/--apply/--verify/--rollback` flows.
- Introduce risk levels and default-safe behavior.
- Validate NVIDIA Container Toolkit and GPU passthrough via real container tests.
- Improve Dockerfile templates to align with detected stack and GPU backend.

**Success criteria**:
- Users are comfortable using `fix --apply` on real machines.
- Docker output works with GPUs consistently or fails with clear guidance.

---

### Phase 4 – Integrations & Community (ongoing)

**Focus**: MCP, CI, adoption flywheel.

- Complete MCP server, publish to MCP catalogs.
- Ship GitHub Action and CI examples.
- Improve docs, README, badges, and trusted publishing setup.
- (Optional) VS Code extension or editor plugins.

**Success criteria**:
- Project used inside AI assistants.
- Adopted in CI pipelines of real users.

---

## 4. Priority Summary

If resources are tight, prioritize in this order:

1. **Causal diagnosis + opinionated `doctor`**.  
2. **Hardware/platform universality (CUDA/ROCm/DirectML/MPS)**.
3. **Safe, explainable `fix` with verification**.
4. **MCP and CI integration**.

Executed well, this plan makes mlenvdoctor a future-proof, community-grade tool that can outgrow current competitors and become the default way to debug ML environments.
