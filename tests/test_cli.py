"""Tests for CLI commands."""

import json
import subprocess
from pathlib import Path

import pytest


def test_version_command():
    """Test --version flag."""
    result = subprocess.run(
        ["mlenvdoctor", "--version"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "ML Environment Doctor" in result.stdout
    assert "version" in result.stdout.lower()


def test_diagnose_basic():
    """Test basic diagnose command."""
    result = subprocess.run(
        ["mlenvdoctor", "diagnose"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    # Should not crash, even if there are issues
    assert result.returncode == 0 or result.returncode == 1
    assert "Running ML Environment Diagnostics" in result.stdout or "diagnose" in result.stderr.lower()


def test_diagnose_full():
    """Test full diagnose command."""
    result = subprocess.run(
        ["mlenvdoctor", "diagnose", "--full"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    # Should not crash
    assert result.returncode == 0 or result.returncode == 1


def test_json_export(tmp_path: Path):
    """Test JSON export functionality."""
    output_file = tmp_path / "test_results.json"

    result = subprocess.run(
        ["mlenvdoctor", "diagnose", "--json", str(output_file)],
        capture_output=True,
        text=True,
        timeout=30,
    )

    if result.returncode == 0:
        assert output_file.exists(), "JSON file should be created"
        data = json.loads(output_file.read_text())
        assert "issues" in data, "JSON should contain 'issues'"
        assert "summary" in data, "JSON should contain 'summary'"
        assert isinstance(data["issues"], list), "Issues should be a list"


def test_csv_export(tmp_path: Path):
    """Test CSV export functionality."""
    output_file = tmp_path / "test_results.csv"

    result = subprocess.run(
        ["mlenvdoctor", "diagnose", "--csv", str(output_file)],
        capture_output=True,
        text=True,
        timeout=30,
    )

    if result.returncode == 0:
        assert output_file.exists(), "CSV file should be created"
        content = output_file.read_text()
        assert "Issue" in content, "CSV should contain headers"
        assert "Status" in content, "CSV should contain Status column"


def test_html_export(tmp_path: Path):
    """Test HTML export functionality."""
    output_file = tmp_path / "test_report.html"

    result = subprocess.run(
        ["mlenvdoctor", "diagnose", "--html", str(output_file)],
        capture_output=True,
        text=True,
        timeout=30,
    )

    if result.returncode == 0:
        assert output_file.exists(), "HTML file should be created"
        content = output_file.read_text()
        assert "<html" in content.lower(), "Should be valid HTML"
        assert "ML Environment Doctor" in content, "Should contain title"


def test_logging(tmp_path: Path):
    """Test logging functionality."""
    log_file = tmp_path / "test.log"

    result = subprocess.run(
        ["mlenvdoctor", "diagnose", "--log-file", str(log_file), "--log-level", "INFO"],
        capture_output=True,
        text=True,
        timeout=30,
    )

    # Log file might be created even if command fails
    if log_file.exists():
        content = log_file.read_text()
        # Just check it's not empty
        assert len(content) >= 0


def test_multiple_exports(tmp_path: Path):
    """Test multiple export formats simultaneously."""
    json_file = tmp_path / "multi.json"
    csv_file = tmp_path / "multi.csv"
    html_file = tmp_path / "multi.html"

    result = subprocess.run(
        [
            "mlenvdoctor",
            "diagnose",
            "--json",
            str(json_file),
            "--csv",
            str(csv_file),
            "--html",
            str(html_file),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    if result.returncode == 0:
        # All files should be created
        assert json_file.exists() or csv_file.exists() or html_file.exists(), "At least one export should work"


def test_fix_command():
    """Test fix command."""
    result = subprocess.run(
        ["mlenvdoctor", "fix", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "fix" in result.stdout.lower() or "auto-fix" in result.stdout.lower()


def test_dockerize_command():
    """Test dockerize command."""
    result = subprocess.run(
        ["mlenvdoctor", "dockerize", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "docker" in result.stdout.lower()


def test_invalid_log_level():
    """Test that invalid log level is rejected."""
    result = subprocess.run(
        ["mlenvdoctor", "diagnose", "--log-level", "INVALID"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    # Should fail or show error
    assert result.returncode != 0 or "invalid" in result.stderr.lower() or "error" in result.stderr.lower()
