"""Dockerfile generation for ML Environment Doctor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import load_config
from .fix import get_stack_requirements
from .utils import console, print_info, print_success


@dataclass(frozen=True)
class DockerModelProfile:
    """Container generation profile for a supported model."""

    slug: str
    model_name: str
    recommended_memory: str
    stack: str


MODEL_PROFILES = {
    "mistral-7b": DockerModelProfile(
        slug="mistral-7b",
        model_name="mistralai/Mistral-7B-v0.1",
        recommended_memory="16GB+",
        stack="llm-training",
    ),
    "tinyllama": DockerModelProfile(
        slug="tinyllama",
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        recommended_memory="4GB+",
        stack="minimal",
    ),
    "gpt2": DockerModelProfile(
        slug="gpt2",
        model_name="gpt2",
        recommended_memory="2GB+",
        stack="minimal",
    ),
}


def _build_pip_install_lines(packages: list[str]) -> str:
    """Render pip install lines for Dockerfiles."""
    rendered = "RUN pip install --no-cache-dir \\\n"
    for index, package in enumerate(packages):
        suffix = " \\\n" if index < len(packages) - 1 else "\n"
        rendered += f"    {package}{suffix}"
    return rendered


def _resolve_base_image(base_image: Optional[str]) -> str:
    """Resolve the base image from args or config."""
    config = load_config()
    configured = str(config.get("docker", {}).get("default_base_image", "")).strip()
    return base_image or configured or "nvidia/cuda:12.4.0-devel-ubuntu22.04"


def generate_dockerfile(
    model_name: Optional[str] = None,
    service: bool = False,
    output_file: str = "Dockerfile.mlenvdoctor",
    *,
    stack: Optional[str] = None,
    base_image: Optional[str] = None,
    python_version: str = "3.10",
    include_healthcheck: bool = True,
) -> Path:
    """Generate a Dockerfile for ML workloads or service deployment."""
    output_path = Path(output_file)
    model_profile = MODEL_PROFILES.get(model_name.lower()) if model_name else None
    if model_name and model_profile is None:
        print_info(f"Unknown model template: {model_name}. Using generic settings.")

    selected_stack = stack or (model_profile.stack if model_profile else "llm-training")
    base = _resolve_base_image(base_image)
    packages = get_stack_requirements(selected_stack)
    pip_install_block = _build_pip_install_lines(packages + ["numpy>=1.24.0", "scipy>=1.10.0"])

    python_pkg = f"python{python_version}"
    dockerfile_content = f"""# ML Environment Doctor - Generated Dockerfile
# Profile: {"service" if service else "training"}
FROM {base}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HOME=/root/.cache/huggingface

RUN apt-get update && apt-get install -y \\
    {python_pkg} \\
    python3-pip \\
    git \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
{pip_install_block}
"""

    if model_profile:
        dockerfile_content += (
            f"\n# Model profile: {model_profile.model_name}\n"
            f"# Recommended GPU memory: {model_profile.recommended_memory}\n"
        )

    if service:
        dockerfile_content += """
RUN pip install --no-cache-dir fastapi uvicorn pydantic

COPY app.py /app/app.py
WORKDIR /app
EXPOSE 8000
"""
        if include_healthcheck:
            dockerfile_content += """
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1
"""
        dockerfile_content += """
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
"""
    else:
        dockerfile_content += """
COPY . /app
WORKDIR /app

RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

CMD ["python", "-c", "print('ML Environment Doctor container is ready. Override CMD with your entrypoint.')"]
"""

    dockerfile_content += """

# Recommended .dockerignore:
# __pycache__/
# *.pyc
# .git/
# .venv/
# *.egg-info/
# .pytest_cache/
# data/
# outputs/
# logs/
"""

    output_path.write_text(dockerfile_content, encoding="utf-8")
    print_success(f"Generated Dockerfile: {output_file}")
    console.print(f"[cyan]Base image: {base}[/cyan]")
    console.print(f"[cyan]Stack: {selected_stack}[/cyan]")
    if model_profile:
        console.print(f"[cyan]Model: {model_profile.model_name}[/cyan]")
        console.print(f"[cyan]Recommended GPU: {model_profile.recommended_memory}[/cyan]")

    console.print()
    console.print("[bold]Build and run:[/bold]")
    console.print(f"[cyan]  docker build -f {output_file} -t mlenvdoctor .[/cyan]")
    if service:
        console.print("[cyan]  docker run --gpus all -p 8000:8000 mlenvdoctor[/cyan]")
    else:
        console.print("[cyan]  docker run --gpus all -v $(pwd)/data:/app/data mlenvdoctor[/cyan]")

    return output_path


def generate_service_template(output_file: str = "app.py") -> Path:
    """Generate a FastAPI service template."""
    output_path = Path(output_file)

    service_content = '''"""FastAPI service template for ML fine-tuning."""
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="ML Fine-tuning Service")


class HealthResponse(BaseModel):
    status: str
    message: str


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(status="healthy", message="Service is running")


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "ML Fine-tuning Service", "version": "0.2.0"}


# Add your inference or fine-tuning endpoints here.

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
'''

    output_path.write_text(service_content, encoding="utf-8")
    print_success(f"Generated service template: {output_file}")
    return output_path
