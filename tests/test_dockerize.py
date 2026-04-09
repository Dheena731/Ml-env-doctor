"""Tests for dockerize module."""

import tempfile
from pathlib import Path

from mlenvdoctor.dockerize import generate_dockerfile, generate_service_template


def test_generate_dockerfile():
    """Test generate_dockerfile function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "Dockerfile"
        result = generate_dockerfile(output_file=str(output_file))
        assert result.exists()
        content = result.read_text()
        assert "FROM" in content
        assert "nvidia/cuda" in content
        assert "torch" in content


def test_generate_dockerfile_with_model():
    """Test generate_dockerfile with model."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "Dockerfile"
        result = generate_dockerfile(model_name="mistral-7b", output_file=str(output_file))
        assert result.exists()
        content = result.read_text()
        assert "mistralai/Mistral-7B" in content or "FROM" in content


def test_generate_dockerfile_service():
    """Test generate_dockerfile with service flag."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "Dockerfile"
        result = generate_dockerfile(service=True, output_file=str(output_file))
        assert result.exists()
        content = result.read_text()
        assert "uvicorn" in content or "FastAPI" in content
        assert "EXPOSE" in content
        assert "COPY requirements.txt" not in content


def test_generate_dockerfile_with_custom_stack_and_base_image():
    """Custom stack and base image options should be reflected in output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "Dockerfile"
        result = generate_dockerfile(
            model_name="tinyllama",
            stack="llm-training",
            base_image="nvidia/cuda:12.4.0-runtime-ubuntu22.04",
            output_file=str(output_file),
        )
        assert result.exists()
        content = result.read_text()
        assert "nvidia/cuda:12.4.0-runtime-ubuntu22.04" in content
        assert "tensorboard>=2.16.0" in content


def test_generate_service_template():
    """Test generate_service_template function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "app.py"
        result = generate_service_template(output_file=str(output_file))
        assert result.exists()
        content = result.read_text()
        assert "FastAPI" in content
        assert "def health" in content
        assert "HTTPException" not in content
