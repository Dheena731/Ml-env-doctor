# Troubleshooting

Use this page when you already have a failing finding and need the shortest practical recovery path.

## PyTorch CUDA Mismatch

**Symptoms**

- `nvidia-smi` works
- `torch.cuda.is_available()` returns `False`
- mlenvdoctor reports `root_pytorch_cuda_mismatch` or `root_pytorch_cpu_only_build`

**Fix**

```bash
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

!!! warning "Common trap"
    Installing `torch` from the default PyPI index may give you a CPU-only or incompatible build depending on platform and version. Use the PyTorch index URL that matches your target CUDA runtime.

## NVIDIA Driver Not Visible

**Symptoms**

- `nvidia-smi` is missing or fails
- Docker GPU tests fail before Python imports matter

**Fix**

=== "Windows / WSL2"

    ```powershell
    nvidia-smi
    wsl --update
    wsl --shutdown
    ```

=== "Linux"

    ```bash
    nvidia-smi
    sudo systemctl restart docker
    ```

Then rerun:

```bash
mlenvdoctor doctor --guided
```

## Pipenv Lock Stale

**Symptoms**

- Your lock file was generated before dependency changes
- The package import differs between shells
- CI and local installs disagree

**Fix**

```bash
pipenv lock --clear
pipenv sync --dev
pipenv run mlenvdoctor diagnose
```

## Conda Environment Drift

**Symptoms**

- `which python` or `where python` points outside the expected environment
- `conda list pytorch` and `python -m pip show torch` disagree

**Fix**

```bash
conda activate ml-debug
python -m pip install --upgrade mlenvdoctor
mlenvdoctor fix --conda --plan
```

## Docker GPU Unavailable

**Symptoms**

- Host `nvidia-smi` works
- `docker run --gpus all ... nvidia-smi` fails
- mlenvdoctor reports `docker_gpu`

**Fix**

```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
mlenvdoctor diagnose --full
```

If the Docker test fails, repair NVIDIA Container Toolkit before changing Python packages.

## Hugging Face Cannot Be Reached

```bash
python -c "import urllib.request; urllib.request.urlopen('https://huggingface.co', timeout=5)"
```

If that fails, check DNS, proxy, VPN, corporate firewall rules, or offline mirrors.

## Still Stuck

Generate a report:

```bash
mlenvdoctor report --output-dir mlenvdoctor-report
```

Share the HTML report with a teammate and attach the JSON report to an issue.
