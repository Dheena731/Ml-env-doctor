# ML Environment Failure Taxonomy

This taxonomy captures the highest-frequency failure patterns for AI developers and beginners.
It is used to keep diagnostics, fix planning, and MCP payloads aligned to practical root causes.

## Priority 1: GPU Runtime Mismatch

- `PT_CPU_WHEEL_ON_GPU`: PyTorch CPU-only wheel installed on a GPU-capable machine.
- `PT_CUDA_RUNTIME_MISMATCH`: PyTorch CUDA build does not match available driver/runtime.
- `DRIVER_NOT_VISIBLE`: NVIDIA tooling is not visible from the active shell/runtime.
- `DOCKER_GPU_RUNTIME_MISMATCH`: Host GPU works, but Docker GPU passthrough fails.

## Priority 2: Platform Path Mismatch

- `WSL_GPU_PASSTHROUGH_MISSING`: WSL environment cannot see host GPU stack correctly.
- `TF_WINDOWS_GPU_PATH`: TensorFlow GPU workflow on Windows is misconfigured for supported path.
- `JAX_CPU_FALLBACK`: JAX backend falls back to CPU unexpectedly.

## Priority 3: Python and Dependency Drift

- `PYTHON_TF_COMPAT_WARNING`: Active Python version may be ahead of stable TensorFlow path for this platform.
- `ML_STACK_INCOMPLETE`: Multiple core ML training dependencies are missing/outdated.

## Priority 4: Beginner-Blocking System Constraints

- `LOW_GPU_MEMORY`: Available VRAM is below practical threshold for target workflows.
- `LOW_DISK_SPACE`: Cache/workspace disk free space is too low for model workflows.

## Product Rules

- Every new diagnostic finding should map to one of these codes when possible.
- Every fix path should reference mismatch codes to keep CLI and MCP behavior consistent.
- New mismatch codes should be additive and backward compatible.
