# Mismatch Codes

Mismatch codes are stable, machine-readable labels that help automation and assistants choose a fix path.

## Priority 1: GPU Runtime Mismatch

- `PT_CPU_WHEEL_ON_GPU`: CPU-only PyTorch wheel installed on a GPU-capable machine.
- `PT_CUDA_RUNTIME_MISMATCH`: PyTorch CUDA build does not match available driver/runtime.
- `DRIVER_NOT_VISIBLE`: NVIDIA tooling is not visible from the active shell/runtime.
- `DOCKER_GPU_RUNTIME_MISMATCH`: Host GPU works, but Docker GPU passthrough fails.

## Priority 2: Platform Path Mismatch

- `WSL_GPU_PASSTHROUGH_MISSING`: WSL environment cannot see the host GPU stack correctly.
- `TF_WINDOWS_GPU_PATH`: TensorFlow GPU workflow on Windows is misconfigured for the supported path.
- `JAX_CPU_FALLBACK`: JAX backend falls back to CPU unexpectedly.

## Priority 3: Python and Dependency Drift

- `PYTHON_TF_COMPAT_WARNING`: Active Python version may be ahead of the stable TensorFlow path for this platform.
- `ML_STACK_INCOMPLETE`: Multiple core ML training dependencies are missing or outdated.

## Priority 4: Beginner-Blocking System Constraints

- `LOW_GPU_MEMORY`: Available VRAM is below the practical threshold for target workflows.
- `LOW_DISK_SPACE`: Cache or workspace disk free space is too low for model workflows.

## Rules for New Codes

- Add codes without changing or reusing existing meanings.
- Map each root-cause diagnostic to one code when possible.
- Include enough evidence in JSON exports for an assistant or CI job to choose the next fix.
