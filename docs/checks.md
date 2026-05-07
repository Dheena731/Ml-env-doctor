# Checks

mlenvdoctor reports 47+ diagnostic signals by combining direct checks, evidence lines, compatibility rules, and root-cause summaries. The stable `check_id` values are what you should key on in automation.

## Status Icons

| Icon | Status | Meaning |
| --- | --- | --- |
| <span class="status-pass">PASS</span> | Healthy | Evidence supports the expected state |
| <span class="status-warn">WARN</span> | Risk | The environment may work, but the path is fragile |
| <span class="status-fail">FAIL</span> | Broken | Fix before training or CI release |

## Diagnostic Matrix

| Group | Signal | `check_id` | What it catches | First fix |
| --- | --- | --- | --- | --- |
| Platform | Python runtime version | `python_runtime` | Unsupported or old interpreter | Use Python 3.8+ |
| Platform | Accelerator backend | `accelerator_backend` | CUDA, CPU, WSL, or macOS backend mismatch | Align runtime with available hardware |
| CUDA | NVIDIA CLI visibility | `cuda_driver` | `nvidia-smi` missing from the active shell | Repair driver visibility |
| CUDA | Driver command failure | `cuda_driver` | Driver installed but failing | Reinstall or restart driver stack |
| CUDA | WSL GPU passthrough | `accelerator_backend` | WSL cannot see host GPU | Update WSL and NVIDIA WSL tooling |
| PyTorch | Import failure | `pytorch_installation` | `import torch` fails | Install PyTorch |
| PyTorch | Version baseline | `pytorch_version` | PyTorch below recommended baseline | Upgrade PyTorch |
| PyTorch | CPU-only build | `pytorch_cuda` | NVIDIA available but `torch.version.cuda` empty | Install CUDA wheel |
| PyTorch | CUDA unavailable | `pytorch_cuda` | `torch.cuda.is_available()` false | Reinstall matching CUDA wheel |
| PyTorch | CUDA execution | `pytorch_cuda_execution` | CUDA imports but tensor operation fails | Recheck runtime and wheel |
| PyTorch | CUDA build metadata | `pytorch_cuda` | Wheel build differs from expected CUDA | Use matching PyTorch index URL |
| PyTorch | GPU count | `pytorch_cuda` | No visible devices | Check driver, container, or permissions |
| PyTorch | Device name evidence | `pytorch_cuda` | Wrong GPU selected | Set device visibility intentionally |
| TensorFlow | TensorFlow import | `tensorflow_runtime` | Missing or broken TensorFlow | Install supported TensorFlow |
| TensorFlow | GPU visibility | `tensorflow_runtime` | TensorFlow falls back to CPU | Use supported GPU path |
| TensorFlow | Tensor execution | `tensorflow_execution` | Import works but op fails | Reinstall runtime |
| TensorFlow | Keras import | `keras_version` | Keras missing with TensorFlow workflow | `pip install keras>=3` |
| JAX | JAX import | `jax_runtime` | Missing JAX | `pip install jax flax` |
| JAX | Backend selection | `jax_runtime` | CPU backend on GPU machine | Install accelerator-enabled JAX |
| JAX | Device enumeration | `jax_runtime` | Device list does not match expectations | Reinstall backend |
| JAX | Array execution | `jax_execution` | Basic JAX operation fails | Reinstall JAX |
| JAX | Flax import | `flax_runtime` | Flax missing for Flax workloads | `pip install flax` |
| ML libs | Transformers | `ml_stack_dependencies` | Missing or old Transformers | Install recommended stack |
| ML libs | Datasets | `ml_stack_dependencies` | Missing or old Datasets | Install recommended stack |
| ML libs | Accelerate | `ml_stack_dependencies` | Missing or old Accelerate | Install recommended stack |
| ML libs | PEFT | `ml_stack_dependencies` | Missing or old PEFT | Install `peft` |
| ML libs | TRL | `ml_stack_dependencies` | Missing or old TRL | Install `trl` |
| ML libs | BitsAndBytes | `ml_stack_dependencies` | Missing optional quantization support | Install compatible build |
| Storage | Cache disk free space | `disk_space` | Model cache drive too small | Free space or move cache |
| GPU | Free GPU memory | `gpu_memory` | Too little memory for training | Close GPU processes |
| Docker | Docker CLI missing | `docker_gpu` | Docker not installed or not in PATH | Install Docker |
| Docker | Docker daemon unavailable | `docker_gpu` | CLI exists but daemon is stopped | Start Docker |
| Docker | GPU passthrough failure | `docker_gpu` | Container cannot use NVIDIA runtime | Install NVIDIA Container Toolkit |
| Network | Hugging Face reachability | `internet_connectivity` | Cannot reach model hub | Check proxy, DNS, or firewall |
| Fix | Requirements generation | `fix_plan` | Needs stack file | Generate requirements |
| Fix | Conda environment file | `fix_plan` | Needs Conda environment | Generate environment YAML |
| Fix | Virtualenv creation | `fix_plan` | Needs isolated environment | Create `.venv` |
| Fix | Rollback snapshot | `fix_plan` | Existing generated files protected | Restore backup if needed |
| Compatibility | TensorFlow on Windows | `compatibility_matrix` | Unsupported native GPU path | Prefer WSL2 or supported backend |
| Compatibility | TensorFlow Python max | `compatibility_matrix` | Python too new for TensorFlow path | Use compatible Python |
| Compatibility | PyTorch/CUDA baseline | `compatibility_matrix` | Wheel and runtime drift | Reinstall matching wheel |
| Root cause | GPU driver unusable | `root_gpu_driver` | Groups driver failures | Repair driver first |
| Root cause | PyTorch missing | `root_pytorch_missing` | Groups import and package evidence | Install PyTorch |
| Root cause | CPU-only PyTorch | `root_pytorch_cpu_only_build` | NVIDIA exists but PyTorch is CPU-only | Install CUDA wheel |
| Root cause | CUDA mismatch | `root_pytorch_cuda_mismatch` | Driver visible but PyTorch cannot use GPU | Reinstall matching wheel |
| Root cause | TensorFlow GPU path | `root_tensorflow_gpu_path` | TensorFlow falls away from GPU | Use supported runtime |
| Root cause | JAX backend | `root_jax_backend` | JAX backend not accelerator-enabled | Install JAX backend |
| Root cause | ML stack dependencies | `root_ml_stack_dependencies` | Core LLM packages missing or old | Generate stack requirements |

## Search Tips

Search for `CUDA`, `pytorch_cuda`, `docker_gpu`, or `root_pytorch_cuda_mismatch` to jump directly to the relevant check.
