"""Tests for CLI commands."""

import json
from pathlib import Path

from typer.testing import CliRunner

from mlenvdoctor.cli import app
from mlenvdoctor.diagnose import DiagnosticIssue

runner = CliRunner()


def _sample_issues() -> list[DiagnosticIssue]:
    return [
        DiagnosticIssue(
            name="Accelerator Backend",
            status="PASS - backend=cuda",
            severity="info",
            fix="",
            check_id="accelerator_backend",
            category="platform",
            metadata={"platform": "linux", "backend": "cuda", "nvidia_visible": True},
        ),
        DiagnosticIssue(
            name="PyTorch CUDA",
            status="FAIL - CUDA not available",
            severity="critical",
            fix="pip install torch --index-url https://download.pytorch.org/whl/cu124",
            check_id="pytorch_cuda",
            category="pytorch",
        ),
        DiagnosticIssue(
            name="transformers",
            status="PASS - 4.44.0",
            severity="info",
            fix="",
            check_id="library_transformers",
            category="dependencies",
        ),
    ]


def test_version_command():
    """Test --version flag."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "ML Environment Doctor" in result.stdout
    assert "version" in result.stdout.lower()


def test_diagnose_basic(monkeypatch):
    """Test basic diagnose command."""
    monkeypatch.setattr("mlenvdoctor.cli.diagnose_env", lambda **kwargs: _sample_issues())
    result = runner.invoke(app, ["diagnose"])
    assert result.exit_code == 2
    assert "Diagnostic Results" in result.stdout


def test_diagnose_full(monkeypatch):
    """Test full diagnose command."""
    monkeypatch.setattr("mlenvdoctor.cli.diagnose_env", lambda **kwargs: _sample_issues())
    monkeypatch.setattr("mlenvdoctor.cli.benchmark_gpu_ops", lambda: {"matmul_1024x1024": 1.2})
    result = runner.invoke(app, ["diagnose", "--full"])
    assert result.exit_code == 2
    assert "GPU benchmark results" in result.stdout


def test_diagnose_json_stdout(monkeypatch):
    """JSON stdout mode should return parseable machine-readable output."""
    monkeypatch.setattr("mlenvdoctor.cli.diagnose_env", lambda **kwargs: _sample_issues())
    result = runner.invoke(app, ["diagnose", "--json", "-"])
    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert "issues" in payload
    assert "doctor_summary" in payload
    assert any(issue["check_id"] == "pytorch_cuda" for issue in payload["issues"])


def test_json_export(monkeypatch, tmp_path: Path):
    """Test JSON export functionality."""
    monkeypatch.setattr("mlenvdoctor.cli.diagnose_env", lambda **kwargs: _sample_issues())
    output_file = tmp_path / "test_results.json"
    result = runner.invoke(app, ["diagnose", "--json", str(output_file)])
    assert result.exit_code == 2
    assert output_file.exists()
    data = json.loads(output_file.read_text())
    assert "issues" in data
    assert "summary" in data


def test_csv_export(monkeypatch, tmp_path: Path):
    """Test CSV export functionality."""
    monkeypatch.setattr("mlenvdoctor.cli.diagnose_env", lambda **kwargs: _sample_issues())
    output_file = tmp_path / "test_results.csv"
    result = runner.invoke(app, ["diagnose", "--csv", str(output_file)])
    assert result.exit_code == 2
    content = output_file.read_text()
    assert "Category" in content
    assert "Check ID" in content


def test_html_export(monkeypatch, tmp_path: Path):
    """Test HTML export functionality."""
    monkeypatch.setattr("mlenvdoctor.cli.diagnose_env", lambda **kwargs: _sample_issues())
    output_file = tmp_path / "test_report.html"
    result = runner.invoke(app, ["diagnose", "--html", str(output_file)])
    assert result.exit_code == 2
    content = output_file.read_text()
    assert "<html" in content.lower()
    assert "ML Environment Doctor" in content


def test_logging(tmp_path: Path):
    """Test logging functionality."""
    log_file = tmp_path / "test.log"
    result = runner.invoke(
        app, ["--log-file", str(log_file), "--log-level", "INFO", "diagnose", "--json", "-"]
    )
    assert result.exit_code in {0, 1, 2}
    assert log_file.exists()


def test_multiple_exports(monkeypatch, tmp_path: Path):
    """Test multiple export formats simultaneously."""
    monkeypatch.setattr("mlenvdoctor.cli.diagnose_env", lambda **kwargs: _sample_issues())
    json_file = tmp_path / "multi.json"
    csv_file = tmp_path / "multi.csv"
    html_file = tmp_path / "multi.html"
    result = runner.invoke(
        app,
        ["diagnose", "--json", str(json_file), "--csv", str(csv_file), "--html", str(html_file)],
    )
    assert result.exit_code == 2
    assert json_file.exists()
    assert csv_file.exists()
    assert html_file.exists()


def test_fix_command():
    """Test fix command."""
    result = runner.invoke(app, ["fix", "--help"])
    assert result.exit_code == 0
    assert "--dry-run" in result.stdout
    assert "--plan" in result.stdout
    assert "--verify" in result.stdout
    assert "--apply" in result.stdout


def test_fix_dry_run(monkeypatch):
    """Dry-run fix mode should not prompt."""

    class Result:
        success = True
        created_paths = []

    monkeypatch.setattr("mlenvdoctor.cli.auto_fix", lambda **kwargs: Result())
    result = runner.invoke(app, ["fix", "--dry-run"])
    assert result.exit_code == 0


def test_dockerize_command():
    """Test dockerize command."""
    result = runner.invoke(app, ["dockerize", "--help"])
    assert result.exit_code == 0
    assert "docker" in result.stdout.lower()


def test_doctor_command(monkeypatch):
    """Test doctor command output."""
    monkeypatch.setattr("mlenvdoctor.cli.diagnose_env", lambda **kwargs: _sample_issues())
    result = runner.invoke(app, ["doctor", "--ci"])
    assert result.exit_code == 2
    assert "mlenvdoctor status=2" in result.stdout
    assert "ISSUE PyTorch CUDA" in result.stdout


def test_doctor_human_output_is_summary_not_table(monkeypatch):
    """Doctor should print a triage summary rather than the diagnose table."""
    monkeypatch.setattr("mlenvdoctor.cli.diagnose_env", lambda **kwargs: _sample_issues())
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 2
    assert "Detected runtime:" in result.stdout
    assert "platform=linux" in result.stdout
    assert "backend=cuda" in result.stdout
    assert "nvidia_tooling=yes" in result.stdout
    assert "Problem: PyTorch CUDA" in result.stdout
    assert "Confidence:" in result.stdout
    assert "Likely cause:" in result.stdout
    assert "Verify:" in result.stdout
    assert "Linked checks:" in result.stdout
    assert "Do this next:" in result.stdout
    assert "Then verify with:" in result.stdout
    assert "Diagnostic Results" not in result.stdout


def test_fix_verify_mode_runs_diagnostics_without_auto_fix(monkeypatch):
    """Verify mode should bypass auto-fix and print triage output."""
    calls = {"auto_fix": 0}

    class Result:
        success = True
        created_paths = []

    def fake_auto_fix(**kwargs):
        calls["auto_fix"] += 1
        return Result()

    monkeypatch.setattr("mlenvdoctor.cli.auto_fix", fake_auto_fix)
    monkeypatch.setattr("mlenvdoctor.cli.diagnose_env", lambda **kwargs: _sample_issues())

    result = runner.invoke(app, ["fix", "--verify"])
    assert result.exit_code == 2
    assert "Doctor Summary" in result.stdout
    assert calls["auto_fix"] == 0


def test_fix_verify_rejects_conflicting_modes():
    """Verify mode should reject apply/plan combinations."""
    result = runner.invoke(app, ["fix", "--verify", "--apply"])
    assert result.exit_code != 0
    assert "--verify cannot be combined" in result.output


def test_stack_command():
    """Test stack command help."""
    result = runner.invoke(app, ["stack", "llm-training", "--help"])
    assert result.exit_code == 0
    assert "llm-training" in result.stdout.lower() or "output" in result.stdout.lower()


def test_invalid_log_level():
    """Test that invalid log level is rejected."""
    result = runner.invoke(app, ["--log-level", "INVALID", "diagnose"])
    assert result.exit_code != 0
    assert "Invalid log level" in result.output or "Invalid value" in result.output


def test_help_survives_log_file_permission_issue(tmp_path: Path):
    """Help output should still work if the log path cannot be opened."""
    blocked_path = tmp_path
    result = runner.invoke(app, ["--log-file", str(blocked_path), "fix", "--help"])
    assert result.exit_code == 0
    assert "fix" in result.stdout.lower()
