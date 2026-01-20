"""Dockerfile generation for ML Environment Doctor."""

from pathlib import Path
from typing import Optional

from rich.console import Console

from .utils import console, print_error, print_info, print_success

# Model-specific templates
MODEL_TEMPLATES = {
    "mistral-7b": {
        "base_image": "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        "model_name": "mistralai/Mistral-7B-v0.1",
        "memory": "16GB+",
    },
    "tinyllama": {
        "base_image": "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "memory": "4GB+",
    },
    "gpt2": {
        "base_image": "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        "model_name": "gpt2",
        "memory": "2GB+",
    },
}


def generate_dockerfile(
    model_name: Optional[str] = None,
    service: bool = False,
    output_file: str = "Dockerfile.mlenvdoctor",
) -> Path:
    """Generate a Dockerfile for ML fine-tuning."""
    output_path = Path(output_file)

    # Get model info if specified
    model_info = None
    if model_name:
        model_info = MODEL_TEMPLATES.get(model_name.lower())
        if not model_info:
            print_info(f"Unknown model template: {model_name}. Using generic template.")

    # Choose base image
    base_image = model_info["base_image"] if model_info else "nvidia/cuda:12.4.0-devel-ubuntu22.04"

    dockerfile_content = f"""# ML Environment Doctor - Generated Dockerfile
# Base image with CUDA support
FROM {base_image}

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HOME=/root/.cache/huggingface

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \\
    python3.10 \\
    python3-pip \\
    git \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.4
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install ML libraries
RUN pip install --no-cache-dir \\
    transformers>=4.44.0 \\
    peft>=0.12.0 \\
    trl>=0.9.0 \\
    datasets>=2.20.0 \\
    accelerate>=1.0.0 \\
    bitsandbytes>=0.43.0 \\
    sentencepiece>=0.1.99 \\
    numpy>=1.24.0 \\
    scipy>=1.10.0

"""

    if service:
        dockerfile_content += """# Install FastAPI and uvicorn for service
RUN pip install --no-cache-dir fastapi uvicorn pydantic

# Copy service code
COPY app.py /app/app.py
COPY requirements.txt /app/requirements.txt
WORKDIR /app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run service
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

"""
    else:
        if model_info:
            dockerfile_content += f"""# Model: {model_info['model_name']}
# Recommended GPU memory: {model_info['memory']}

"""
        dockerfile_content += """# Copy application code
COPY . /app
WORKDIR /app

# Install additional dependencies if requirements.txt exists
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Default command (override in docker run)
CMD ["python", "train.py"]

"""

    # Add .dockerignore suggestion comment
    dockerfile_content += """# Recommended .dockerignore:
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

    if model_info:
        console.print(f"[cyan]Model: {model_info['model_name']}[/cyan]")
        console.print(f"[cyan]Recommended GPU: {model_info['memory']}[/cyan]")

    console.print()
    console.print("[bold]Build and run:[/bold]")
    console.print(f"[cyan]  docker build -f {output_file} -t mlenvdoctor .[/cyan]")
    if service:
        console.print(f"[cyan]  docker run --gpus all -p 8000:8000 mlenvdoctor[/cyan]")
    else:
        console.print(f"[cyan]  docker run --gpus all -v $(pwd)/data:/app/data mlenvdoctor[/cyan]")

    return output_path


def generate_service_template(output_file: str = "app.py") -> Path:
    """Generate a FastAPI service template."""
    output_path = Path(output_file)

    service_content = '''"""FastAPI service template for ML fine-tuning."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

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
    return {"message": "ML Fine-tuning Service", "version": "0.1.0"}


# Add your fine-tuning endpoints here
# Example:
# @app.post("/fine-tune")
# async def fine_tune(model_name: str, dataset_path: str):
#     ...


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

'''

    output_path.write_text(service_content, encoding="utf-8")
    print_success(f"Generated service template: {output_file}")
    return output_path
