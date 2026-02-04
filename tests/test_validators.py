"""Tests for validators module."""

import pytest
from pathlib import Path

from mlenvdoctor.exceptions import ConfigurationError
from mlenvdoctor.validators import (
    validate_file_path,
    validate_log_level,
    validate_model_name,
    validate_stack_name,
    validate_timeout,
    sanitize_command,
)


def test_validate_model_name_valid():
    """Test valid model names."""
    assert validate_model_name("tinyllama") == "tinyllama"
    assert validate_model_name("GPT2") == "gpt2"
    assert validate_model_name("mistral-7b") == "mistral-7b"
    assert validate_model_name("model_name") == "model_name"


def test_validate_model_name_invalid():
    """Test invalid model names."""
    with pytest.raises(ConfigurationError):
        validate_model_name("")

    with pytest.raises(ConfigurationError):
        validate_model_name("model; rm -rf /")

    with pytest.raises(ConfigurationError):
        validate_model_name("model && ls")


def test_validate_log_level_valid():
    """Test valid log levels."""
    assert validate_log_level("DEBUG") == "DEBUG"
    assert validate_log_level("info") == "INFO"
    assert validate_log_level("Warning") == "WARNING"


def test_validate_log_level_invalid():
    """Test invalid log levels."""
    with pytest.raises(ConfigurationError):
        validate_log_level("INVALID")

    with pytest.raises(ConfigurationError):
        validate_log_level("debugging")


def test_validate_stack_name_valid():
    """Test valid stack names."""
    assert validate_stack_name("trl-peft") == "trl-peft"
    assert validate_stack_name("MINIMAL") == "minimal"


def test_validate_stack_name_invalid():
    """Test invalid stack names."""
    with pytest.raises(ConfigurationError):
        validate_stack_name("invalid-stack")

    with pytest.raises(ConfigurationError):
        validate_stack_name("custom")


def test_validate_timeout_valid():
    """Test valid timeout values."""
    assert validate_timeout(10) == 10
    assert validate_timeout(100) == 100
    assert validate_timeout(None) is None


def test_validate_timeout_invalid():
    """Test invalid timeout values."""
    with pytest.raises(ConfigurationError):
        validate_timeout(-1)

    with pytest.raises(ConfigurationError):
        validate_timeout(0)

    with pytest.raises(ConfigurationError):
        validate_timeout(10000)  # Exceeds max


def test_sanitize_command_valid():
    """Test valid commands."""
    assert sanitize_command(["ls", "-la"]) == ["ls", "-la"]
    assert sanitize_command(["python", "script.py"]) == ["python", "script.py"]


def test_sanitize_command_invalid():
    """Test invalid commands."""
    with pytest.raises(ConfigurationError):
        sanitize_command(["ls", ";", "rm", "-rf"])

    with pytest.raises(ConfigurationError):
        sanitize_command(["echo", "test && ls"])

    with pytest.raises(ConfigurationError):
        sanitize_command([])

    with pytest.raises(ConfigurationError):
        sanitize_command([123])  # Not a string


def test_validate_file_path(tmp_path: Path):
    """Test file path validation."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("test")

    # Valid existing file
    assert validate_file_path(test_file, must_exist=True, must_be_file=True) == test_file.resolve()

    # Valid non-existent path
    new_path = tmp_path / "new.txt"
    assert validate_file_path(new_path, must_exist=False) == new_path.resolve()

    # Invalid: must exist but doesn't
    with pytest.raises(ConfigurationError):
        validate_file_path(new_path, must_exist=True)

    # Invalid: must be file but is directory
    with pytest.raises(ConfigurationError):
        validate_file_path(tmp_path, must_be_file=True)
