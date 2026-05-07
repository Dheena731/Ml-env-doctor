# Mismatch codes

Mismatch codes are stable, machine-readable labels that help automation and assistants choose a fix path.

Authoritative list: see [Failure taxonomy](../FAILURE_TAXONOMY.md).

Common codes you may see in JSON exports:

- `PT_CPU_WHEEL_ON_GPU`: CPU-only PyTorch wheel installed on a GPU-capable machine.
- `PT_CUDA_RUNTIME_MISMATCH`: CUDA-enabled PyTorch build cannot use local driver/runtime.
- `PYTHON_TF_COMPAT_WARNING`: Python version may be ahead of the stable TensorFlow path on the current platform.

