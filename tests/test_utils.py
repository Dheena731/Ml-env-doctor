"""Tests for utils module."""

import sys

import pytest

from mlenvdoctor.utils import (
    check_command_exists,
    format_size,
    get_home_config_dir,
    get_python_version,
)


def test_format_size():
    """Test format_size function."""
    assert format_size(1024) == "1.00 KB"
    assert format_size(1024 * 1024) == "1.00 MB"
    assert format_size(1024 * 1024 * 1024) == "1.00 GB"


def test_get_python_version():
    """Test get_python_version function."""
    version = get_python_version()
    assert isinstance(version, tuple)
    assert len(version) == 3
    assert all(isinstance(v, int) for v in version)


def test_get_home_config_dir():
    """Test get_home_config_dir function."""
    config_dir = get_home_config_dir()
    assert config_dir.exists()
    assert config_dir.name == ".mlenvdoctor"


def test_check_command_exists():
    """Test check_command_exists function."""
    # Python should exist
    assert check_command_exists("python") or check_command_exists("python3")
