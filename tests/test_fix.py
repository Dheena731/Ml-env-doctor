"""Tests for fix module."""

import tempfile
from pathlib import Path

import pytest

from mlenvdoctor import fix
from mlenvdoctor.diagnose import DiagnosticIssue
from mlenvdoctor.fix import (
    auto_fix,
    generate_conda_env,
    generate_requirements_txt,
    get_stack_requirements,
    install_requirements,
    plan_fixes,
    recommend_stack_from_findings,
)


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


def test_install_requirements_uses_requested_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Install path should honor an explicitly selected Python executable."""
    req_file = tmp_path / "requirements.txt"
    req_file.write_text("transformers>=4.44.0\n", encoding="utf-8")

    commands = []

    def fake_run_command(cmd, timeout=30, **kwargs):
        commands.append(cmd)

        class Result:
            returncode = 0
            stderr = ""

        return Result()

    monkeypatch.setattr(fix, "run_command", fake_run_command)

    success = install_requirements(
        str(req_file),
        python_executable="/custom/venv/python",
    )

    assert success is True
    assert commands, "Expected pip install commands to be issued"
    assert all(command[0] == "/custom/venv/python" for command in commands)


def test_install_requirements_bootstraps_pip_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Install path should try ensurepip before failing when pip is missing."""
    req_file = tmp_path / "requirements.txt"
    req_file.write_text("transformers>=4.44.0\n", encoding="utf-8")

    commands = []

    def fake_run_command(cmd, timeout=30, **kwargs):
        commands.append(cmd)

        class Result:
            def __init__(self, returncode: int, stderr: str = ""):
                self.returncode = returncode
                self.stderr = stderr

        if cmd[2:4] == ["pip", "--version"]:
            return Result(1, "No module named pip")
        return Result(0)

    monkeypatch.setattr(fix, "run_command", fake_run_command)

    success = install_requirements(str(req_file), python_executable="/custom/venv/python")

    assert success is True
    assert ["/custom/venv/python", "-m", "ensurepip", "--upgrade"] in commands
    assert ["/custom/venv/python", "-m", "pip", "install", "-r", str(req_file)] in commands


def test_llm_training_stack_contains_training_packages():
    """The llm-training stack should include core fine-tuning dependencies."""
    requirements = get_stack_requirements("llm-training")

    assert "torch>=2.4.0" in requirements
    assert "trl>=0.9.0" in requirements
    assert "tensorboard>=2.16.0" in requirements


def test_plan_fixes_creates_requirements_action_for_critical_dependency_issue():
    """Critical dependency issues should produce a requirements-file action."""
    issues = [
        DiagnosticIssue(
            name="PyTorch CUDA",
            status="FAIL - CUDA not available",
            severity="critical",
            fix="pip install torch",
            check_id="pytorch_cuda",
            category="pytorch",
        )
    ]

    actions = plan_fixes(issues, use_conda=False, create_venv=False, stack="trl-peft")

    assert any(action.kind == "write_requirements" for action in actions)


def test_plan_fixes_uses_grouped_root_causes_for_tensorflow_and_jax():
    """Fix planning should follow grouped doctor findings, not only raw critical issue types."""
    issues = [
        DiagnosticIssue(
            name="TensorFlow",
            status="WARN - 2.16.1 (GPU not visible)",
            severity="warning",
            fix="pip install --upgrade tensorflow[and-cuda]",
            check_id="tensorflow_runtime",
            category="tensorflow",
        ),
        DiagnosticIssue(
            name="JAX",
            status="WARN - 0.4.30 (CPU backend)",
            severity="warning",
            fix='pip install --upgrade "jax[cuda]"',
            check_id="jax_runtime",
            category="jax",
        ),
    ]

    actions = plan_fixes(issues, use_conda=False, create_venv=False, stack="llm-training")

    assert any(action.kind == "write_requirements" for action in actions)


def test_recommend_stack_from_findings_prefers_minimal_for_pytorch_runtime_repairs():
    """Grouped PyTorch runtime issues should prefer the minimal repair stack."""
    findings = [
        fix.DoctorFinding(
            problem="PyTorch cannot use the available GPU",
            severity="critical",
            confidence="high",
            likely_cause="mismatch",
            best_fix="repair torch",
            verify_steps=["verify"],
            evidence=["evidence"],
            check_id="root_pytorch_cuda_mismatch",
            category="pytorch",
        )
    ]

    assert recommend_stack_from_findings("trl-peft", findings) == "minimal"


def test_recommend_stack_from_findings_prefers_llm_training_for_dependency_stack_repairs():
    """Grouped dependency stack issues should prefer the full llm-training stack."""
    findings = [
        fix.DoctorFinding(
            problem="The core ML training dependency stack is incomplete or outdated",
            severity="critical",
            confidence="high",
            likely_cause="deps",
            best_fix="refresh stack",
            verify_steps=["verify"],
            evidence=["transformers"],
            check_id="root_ml_stack_dependencies",
            category="dependencies",
        )
    ]

    assert recommend_stack_from_findings("trl-peft", findings) == "llm-training"


def test_auto_fix_dry_run_returns_actions(monkeypatch: pytest.MonkeyPatch):
    """Dry-run auto-fix should return the planned actions without executing them."""
    issues = [
        DiagnosticIssue(
            name="NVIDIA GPU Driver",
            status="PASS - CUDA 12.4",
            severity="info",
            fix="",
            check_id="cuda_driver",
            category="gpu",
        ),
        DiagnosticIssue(
            name="PyTorch CUDA",
            status="FAIL - CUDA not available",
            severity="critical",
            fix="pip install torch",
            check_id="pytorch_cuda",
            category="pytorch",
        ),
    ]

    monkeypatch.setattr(fix, "diagnose_env", lambda **kwargs: issues)

    result = auto_fix(dry_run=True)

    assert result.success is True
    assert result.actions
    assert result.created_paths == []
    assert result.reasons
    assert result.selected_stack == "minimal"


def test_auto_fix_verification_summary(monkeypatch: pytest.MonkeyPatch):
    """Successful apply should return a before/after verification summary."""
    before_issues = [
        DiagnosticIssue(
            name="PyTorch CUDA",
            status="FAIL - CUDA not available",
            severity="critical",
            fix="pip install torch",
            check_id="pytorch_cuda",
            category="pytorch",
        ),
        DiagnosticIssue(
            name="Disk Space",
            status="WARN - Low space",
            severity="warning",
            fix="free disk space",
            check_id="disk_space",
            category="system",
        ),
    ]
    after_issues = [
        DiagnosticIssue(
            name="Disk Space",
            status="WARN - Low space",
            severity="warning",
            fix="free disk space",
            check_id="disk_space",
            category="system",
        )
    ]

    calls = {"count": 0}

    def fake_diagnose_env(**kwargs):
        calls["count"] += 1
        return before_issues if calls["count"] == 1 else after_issues

    monkeypatch.setattr(fix, "diagnose_env", fake_diagnose_env)
    monkeypatch.setattr(
        fix,
        "_execute_action",
        lambda *args, **kwargs: (True, Path("requirements-mlenvdoctor.txt"), "ok"),
    )

    result = auto_fix(apply=True, verify=True)

    assert result.success is True
    assert result.verification_summary is not None
    assert result.verification_summary.before_critical == 1
    assert result.verification_summary.after_critical == 0
    assert "PyTorch CUDA" in result.verification_summary.resolved_critical


def test_auto_fix_uses_doctor_findings_as_reasons(monkeypatch: pytest.MonkeyPatch):
    """Fix results should expose the user-facing reasons behind planned actions."""
    issues = [
        DiagnosticIssue(
            name="transformers",
            status="FAIL - Not installed",
            severity="critical",
            fix="pip install transformers>=4.44.0",
            check_id="library_transformers",
            category="dependencies",
        ),
        DiagnosticIssue(
            name="accelerate",
            status="WARN - Old version (0.9.0)",
            severity="warning",
            fix="pip install --upgrade accelerate>=1.0.0",
            check_id="library_accelerate",
            category="dependencies",
        ),
    ]

    monkeypatch.setattr(fix, "diagnose_env", lambda **kwargs: issues)

    result = auto_fix(dry_run=True)

    assert any("dependency stack" in reason.lower() for reason in result.reasons)
