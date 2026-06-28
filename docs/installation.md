# Installation

Install mlenvdoctor in the same Python environment where your ML stack runs. Most users should start with `pip`.

## pip

=== "Windows PowerShell"

    ```powershell
    py -m pip install --upgrade pip
    py -m pip install mlenvdoctor
    mlenvdoctor --version
    ```

=== "macOS / Linux"

    ```bash
    python -m pip install --upgrade pip
    python -m pip install mlenvdoctor
    mlenvdoctor --version
    ```

=== "Conda"

    ```bash
    conda create -n ml-debug python=3.11 -y
    conda activate ml-debug
    python -m pip install mlenvdoctor
    mlenvdoctor doctor --guided
    ```

!!! warning "Install into the active environment"
    If your model runs inside `.venv`, Conda, Pipenv, WSL, or a container, install and run mlenvdoctor there. Running it from a different shell diagnoses the wrong machine state.

## Docker Workflow

Use `dockerize` when you want a reproducible starting point for model testing or fine-tuning.

```bash
mlenvdoctor dockerize tinyllama --stack minimal
docker build -f Dockerfile.mlenvdoctor -t tinyllama-env .
```

For GPU containers, verify the host first:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
mlenvdoctor diagnose --full
```

## Development Setup

```bash
git clone https://github.com/Dheena731/Ml-env-doctor.git
cd Ml-env-doctor
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
pre-commit install
pytest
```

=== "Windows activation"

    ```powershell
    .\.venv\Scripts\Activate.ps1
    ```

=== "macOS / Linux activation"

    ```bash
    source .venv/bin/activate
    ```

## Documentation Setup

```bash
python -m pip install -r requirements-docs.txt
mkdocs serve
```

`requirements-docs.txt` installs the project `docs` extra, so documentation dependencies stay aligned with `pyproject.toml`.

The local preview opens at `http://127.0.0.1:8000/`.
