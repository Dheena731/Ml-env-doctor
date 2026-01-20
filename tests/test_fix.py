"""Tests for fix module."""

import tempfile
from pathlib import Path

import pytest

from mlenvdoctor.fix import generate_conda_env, generate_requirements_txt


def test_generate_requirements_txt():
    """Test generate_requirements_txt function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "requirements.txt"
        result = generate_requirements_txt(stack="trl-peft", output_file=str(output_file))
        assert result.exists()
        content = result.read_text()
        assert "torch" in content
        assert "transformers" in content
        assert "peft" in content


def test_generate_requirements_txt_minimal():
    """Test generate_requirements_txt with minimal stack."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "requirements.txt"
        result = generate_requirements_txt(stack="minimal", output_file=str(output_file))
        assert result.exists()
        content = result.read_text()
        assert "torch" in content
        assert "transformers" in content


def test_generate_conda_env():
    """Test generate_conda_env function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "environment.yml"
        result = generate_conda_env(stack="trl-peft", output_file=str(output_file))
        assert result.exists()
        content = result.read_text()
        assert "name: mlenvdoctor" in content
        assert "pytorch" in content
        assert "pip:" in content


def test_generate_requirements_invalid_stack():
    """Test generate_requirements_txt with invalid stack."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "requirements.txt"
        with pytest.raises(SystemExit):
            generate_requirements_txt(stack="invalid", output_file=str(output_file))
