"""Tests for utils module."""

from pathlib import Path

from mlenvdoctor.diagnose import DiagnosticIssue
from mlenvdoctor.export import build_export_data, export_html, get_exit_code
from mlenvdoctor.mcp import _handle_request
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


def test_export_html_escapes_issue_content(tmp_path: Path):
    """HTML export should escape diagnostic fields."""
    output_file = tmp_path / "report.html"
    issues = [
        DiagnosticIssue(
            name="<script>alert(1)</script>",
            status="FAIL - <broken>",
            severity="warning",
            fix="<b>fix</b>",
            details="<img src=x>",
        )
    ]

    export_html(issues, output_file)
    content = output_file.read_text(encoding="utf-8")

    assert "<script>alert(1)</script>" not in content
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in content


def test_build_export_data_includes_exit_code():
    """Machine-readable export data should include the diagnostic exit code."""
    issues = [DiagnosticIssue("A", "WARN - attention", "warning", "pip install a")]

    data = build_export_data(issues, include_metadata=False)

    assert data["exit_code"] == 1
    assert data["summary"]["warnings"] == 1


def test_get_exit_code_prefers_critical_over_warning():
    """Critical issues should map to exit code 2."""
    issues = [
        DiagnosticIssue("A", "WARN - attention", "warning", "pip install a"),
        DiagnosticIssue("B", "FAIL - broken", "critical", "pip install b"),
    ]

    assert get_exit_code(issues) == 2


def test_mcp_stub_reports_unknown_tool():
    """The MCP stub should provide a stable error shape for unknown tools."""
    response = _handle_request({"tool": "missing", "arguments": {}})

    assert response["ok"] is False
    assert "available_tools" in response


def test_mcp_stub_get_fixes_uses_machine_readable_shape(monkeypatch):
    """The MCP stub should return fix data in a stable schema."""
    issues = [DiagnosticIssue("Torch", "FAIL - missing", "critical", "pip install torch")]

    monkeypatch.setattr("mlenvdoctor.mcp.diagnose_env", lambda full=False, show_header=False: issues)

    response = _handle_request({"tool": "get_fixes", "arguments": {"full": False}})

    assert response["ok"] is True
    assert response["result"]["exit_code"] == 2
    assert response["result"]["fixes"][0]["command"] == "pip install torch"
