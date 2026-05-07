# Troubleshooting

## “CUDA not available” but I have an NVIDIA GPU

Start with:

```bash
mlenvdoctor doctor --guided
mlenvdoctor diagnose --json -
```

Look for mismatch codes like `PT_CPU_WHEEL_ON_GPU` or `PT_CUDA_RUNTIME_MISMATCH`.

## WSL2: GPU not visible inside WSL

- Verify GPU works on Windows host first.
- Then inside WSL:

```bash
nvidia-smi
mlenvdoctor doctor
```

## TensorFlow on Windows

TensorFlow GPU support paths differ across platforms. If you are blocked, prefer WSL2 for CUDA-based workflows.

## Docker GPU passthrough fails

Run:

```bash
mlenvdoctor diagnose --full
```

This includes a Docker GPU test and points to common `nvidia-container-toolkit` runtime issues.

