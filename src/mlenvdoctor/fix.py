"""Auto-fix and requirements generation for ML Environment Doctor."""

from __future__ import annotations

import sys
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .config import load_config
from .constants import DEFAULT_REQUIREMENTS_FILE
from .diagnose import DiagnosticIssue, DoctorFinding, diagnose_env, summarize_for_doctor
from .icons import icon_wrench
from .utils import (
    check_command_exists,
    console,
    print_error,
    print_info,
    print_success,
    print_warning,
    run_command,
    status_message,
)

ML_STACKS = {
    "trl-peft": [
        "torch>=2.4.0",
        "transformers>=4.44.0",
        "peft>=0.12.0",
        "trl>=0.9.0",
        "datasets>=2.20.0",
        "accelerate>=1.0.0",
        "bitsandbytes>=0.43.0",
        "sentencepiece>=0.1.99",
    ],
    "minimal": [
        "torch>=2.4.0",
        "transformers>=4.44.0",
        "accelerate>=1.0.0",
    ],
    "llm-training": [
        "torch>=2.4.0",
        "transformers>=4.44.0",
        "datasets>=2.20.0",
        "accelerate>=1.0.0",
        "peft>=0.12.0",
        "trl>=0.9.0",
        "bitsandbytes>=0.43.0",
        "sentencepiece>=0.1.99",
        "safetensors>=0.4.3",
        "tensorboard>=2.16.0",
    ],
}


@dataclass
class FixAction:
    """A validated action the fix engine can perform."""

    kind: str
    description: str
    risk: str = "low"
    command: Optional[List[str]] = None
    output_path: Optional[Path] = None
    details: str = ""


@dataclass
class FixResult:
    """Structured result for a fix run."""

    success: bool
    actions: List[FixAction] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    selected_stack: str = "trl-peft"
    executed_actions: List[str] = field(default_factory=list)
    verification_issues: List[DiagnosticIssue] = field(default_factory=list)
    verification_summary: Optional["FixVerificationSummary"] = None
    created_paths: List[Path] = field(default_factory=list)
    message: str = ""


@dataclass
class FixVerificationSummary:
    """Summarize before/after diagnostic state for a fix attempt."""

    before_critical: int
    after_critical: int
    before_warning: int
    after_warning: int
    resolved_critical: List[str] = field(default_factory=list)
    remaining_critical: List[str] = field(default_factory=list)
    remaining_warning: List[str] = field(default_factory=list)
    confidence_delta: int = 0


def _count_issues(issues: List[DiagnosticIssue], *, severity: str) -> int:
    """Count failing or warning issues for a given severity."""
    if severity == "critical":
        return sum(1 for issue in issues if issue.severity == "critical" and "FAIL" in issue.status)
    return sum(
        1
        for issue in issues
        if issue.severity == "warning" and issue.status.startswith(("WARN", "FAIL"))
    )


def _risk_score(critical: int, warning: int) -> int:
    """Weighted issue score used for verification confidence change."""
    return critical * 2 + warning


def _build_verification_summary(
    before_issues: List[DiagnosticIssue], after_issues: List[DiagnosticIssue]
) -> FixVerificationSummary:
    """Create a before/after verification summary."""
    before_critical_names = {
        issue.name
        for issue in before_issues
        if issue.severity == "critical" and "FAIL" in issue.status
    }
    after_critical_names = {
        issue.name
        for issue in after_issues
        if issue.severity == "critical" and "FAIL" in issue.status
    }
    after_warning_names = [
        issue.name
        for issue in after_issues
        if issue.severity == "warning" and issue.status.startswith(("WARN", "FAIL"))
    ]

    before_critical = _count_issues(before_issues, severity="critical")
    after_critical = _count_issues(after_issues, severity="critical")
    before_warning = _count_issues(before_issues, severity="warning")
    after_warning = _count_issues(after_issues, severity="warning")

    return FixVerificationSummary(
        before_critical=before_critical,
        after_critical=after_critical,
        before_warning=before_warning,
        after_warning=after_warning,
        resolved_critical=sorted(before_critical_names - after_critical_names),
        remaining_critical=sorted(after_critical_names),
        remaining_warning=after_warning_names,
        confidence_delta=_risk_score(before_critical, before_warning)
        - _risk_score(after_critical, after_warning),
    )


def recommend_stack_from_findings(
    requested_stack: str,
    findings: List[DoctorFinding],
) -> str:
    """Choose a practical stack based on grouped root causes."""
    if requested_stack != "trl-peft":
        return requested_stack

    finding_ids = {finding.check_id for finding in findings}
    if "root_ml_stack_dependencies" in finding_ids:
        return "llm-training"
    if {
        "root_pytorch_missing",
        "root_pytorch_cpu_only_build",
        "root_pytorch_cuda_mismatch",
    } & finding_ids:
        return "minimal"
    return requested_stack


def get_stack_requirements(stack: str) -> list[str]:
    """Return the requirement list for a named stack."""
    if stack not in ML_STACKS:
        print_error(f"Unknown stack: {stack}. Available: {list(ML_STACKS.keys())}")
        sys.exit(1)
    return ML_STACKS[stack]


def generate_requirements_txt(
    stack: str = "trl-peft", output_file: str = DEFAULT_REQUIREMENTS_FILE
) -> Path:
    """Generate a requirements.txt file for a named stack."""
    requirements = get_stack_requirements(stack)
    output_path = Path(output_file)

    try:
        import torch

        if torch.cuda.is_available():
            content = "# Install CUDA-enabled PyTorch first if needed\n"
            content += (
                "# pip install torch --index-url " "https://download.pytorch.org/whl/cu124\n\n"
            )
        else:
            content = "# Install the appropriate PyTorch build for your machine\n\n"
    except ImportError:
        content = "# Install the appropriate PyTorch build for your machine\n"
        content += "# CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu124\n"
        content += "# CPU: pip install torch\n\n"

    content += "\n".join(requirements)
    content += "\n"

    output_path.write_text(content, encoding="utf-8")
    print_success(f"Generated {output_path}")
    return output_path


def generate_conda_env(
    stack: str = "trl-peft", output_file: str = "environment-mlenvdoctor.yml"
) -> Path:
    """Generate a conda environment file."""
    requirements = get_stack_requirements(stack)
    output_path = Path(output_file)

    content = "name: mlenvdoctor\n"
    content += "channels:\n"
    content += "  - pytorch\n"
    content += "  - nvidia\n"
    content += "  - conda-forge\n"
    content += "  - defaults\n"
    content += "dependencies:\n"
    content += "  - python>=3.8\n"
    content += "  - pytorch>=2.4.0\n"
    content += "  - pytorch-cuda=12.4\n"
    content += "  - pip\n"
    content += "  - pip:\n"

    pip_requirements = [req for req in requirements if not req.startswith("torch")]
    for req in pip_requirements:
        content += f"    - {req}\n"

    output_path.write_text(content, encoding="utf-8")
    print_success(f"Generated {output_path}")
    print_info(f"Create environment with: conda env create -f {output_file}")
    return output_path


def get_virtualenv_python(env_path: Path) -> Path:
    """Return the Python executable inside a virtual environment."""
    if sys.platform == "win32":
        return env_path / "Scripts" / "python.exe"
    return env_path / "bin" / "python"


def install_requirements(
    requirements_file: str,
    use_conda: bool = False,
    python_executable: Optional[str] = None,
) -> bool:
    """Install requirements from file."""
    req_path = Path(requirements_file)
    if not req_path.exists():
        print_error(f"Requirements file not found: {requirements_file}")
        return False

    if use_conda:
        print_info("Using conda for installation...")
        if not check_command_exists("conda"):
            print_error("conda not found. Install Miniconda/Anaconda first.")
            return False
        print_warning("Conda environment file generated. Please create environment manually.")
        return True

    python_cmd = python_executable or sys.executable
    print_info(f"Installing requirements from {requirements_file}...")

    try:
        pip_check = run_command([python_cmd, "-m", "pip", "--version"], timeout=60)
        if pip_check.returncode != 0:
            print_warning(
                f"pip is not available in {python_cmd}. Attempting to bootstrap it with ensurepip..."
            )
            ensurepip_result = run_command(
                [python_cmd, "-m", "ensurepip", "--upgrade"], timeout=300
            )
            if ensurepip_result.returncode != 0:
                print_error(
                    "Unable to bootstrap pip automatically. "
                    "Install pip in this environment or rerun with a Python that already has pip."
                )
                if ensurepip_result.stderr:
                    print_error(ensurepip_result.stderr.strip())
                return False

        with status_message("[bold green]Installing requirements...[/bold green]"):
            result = run_command(
                [python_cmd, "-m", "pip", "install", "-r", requirements_file],
                timeout=600,
            )
        if result.returncode == 0:
            print_success("Requirements installed successfully!")
            return True

        print_error(f"Installation failed: {result.stderr}")
        return False
    except Exception as exc:
        print_error(f"Installation error: {exc}")
        return False


def create_virtualenv(env_name: str = ".venv") -> Optional[Path]:
    """Create a virtual environment."""
    env_path = Path(env_name)
    if env_path.exists():
        print_warning(f"Virtual environment already exists: {env_name}")
        return env_path

    print_info(f"Creating virtual environment: {env_name}...")
    try:
        import venv

        venv.create(env_path, with_pip=True)
        print_success(f"Virtual environment created: {env_name}")
        if sys.platform == "win32":
            activate_cmd = rf"{env_name}\Scripts\activate"
        else:
            activate_cmd = f"source {env_name}/bin/activate"
        print_info(f"Activate with: {activate_cmd}")
        return env_path
    except Exception as exc:
        print_error(f"Failed to create virtual environment: {exc}")
        return None


def _backup_root() -> Path:
    """Return the fix backup root directory."""
    return Path(".mlenvdoctor") / "fix-backups"


def create_fix_backup(paths: List[Path]) -> Optional[Path]:
    """Create a backup snapshot for rollback-safe file operations."""
    tracked_paths = [path for path in paths if path.exists() and path.is_file()]
    if not tracked_paths:
        return None

    backup_dir = _backup_root() / datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_dir.mkdir(parents=True, exist_ok=True)
    manifest_lines: List[str] = []
    for path in tracked_paths:
        relative_name = path.as_posix().replace("/", "__")
        target_path = backup_dir / relative_name
        target_path.write_bytes(path.read_bytes())
        manifest_lines.append(f"{relative_name}\t{path.as_posix()}")

    (backup_dir / "manifest.tsv").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    return backup_dir


def rollback_last_fix() -> Dict[str, object]:
    """Restore files from the latest fix backup snapshot."""
    root = _backup_root()
    if not root.exists():
        return {"ok": False, "message": "No fix backups found."}

    candidates = sorted([path for path in root.iterdir() if path.is_dir()])
    if not candidates:
        return {"ok": False, "message": "No fix backups found."}

    latest = candidates[-1]
    manifest_path = latest / "manifest.tsv"
    if not manifest_path.exists():
        return {"ok": False, "message": f"Backup manifest missing in {latest}."}

    restored: List[str] = []
    for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        backup_name, original_path = raw_line.split("\t", 1)
        source = latest / backup_name
        target = Path(original_path)
        if source.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
            restored.append(original_path)

    return {
        "ok": True,
        "message": f"Restored {len(restored)} file(s) from backup {latest}.",
        "restored": restored,
    }


def plan_fixes(
    issues: List[DiagnosticIssue],
    *,
    use_conda: bool,
    create_venv: bool,
    stack: str,
    doctor_findings: Optional[List[DoctorFinding]] = None,
    requirements_output: str = DEFAULT_REQUIREMENTS_FILE,
    conda_output: str = "environment-mlenvdoctor.yml",
    venv_path: str = ".venv",
) -> List[FixAction]:
    """Turn diagnostics into a predictable fix plan."""
    findings = doctor_findings if doctor_findings is not None else summarize_for_doctor(issues)
    critical_issues = [
        issue for issue in issues if issue.severity == "critical" and "FAIL" in issue.status
    ]
    actionable_library_issue = any(
        issue.category in {"dependencies", "pytorch"} for issue in critical_issues
    )
    fixable_root_causes = {
        "root_pytorch_missing",
        "root_pytorch_cpu_only_build",
        "root_pytorch_cuda_mismatch",
        "root_ml_stack_dependencies",
        "root_tensorflow_gpu_path",
        "root_jax_backend",
    }
    root_cause_requires_requirements = any(
        finding.check_id in fixable_root_causes
        or finding.category in {"dependencies", "pytorch", "tensorflow", "jax"}
        for finding in findings
    )

    actions: List[FixAction] = []
    if create_venv:
        actions.append(
            FixAction(
                kind="create_venv",
                description=f"Create virtual environment at {venv_path}",
                risk="low",
                output_path=Path(venv_path),
            )
        )

    if use_conda:
        actions.append(
            FixAction(
                kind="write_conda",
                description=f"Generate Conda environment file for stack '{stack}'",
                risk="low",
                output_path=Path(conda_output),
            )
        )
        return actions

    if root_cause_requires_requirements or critical_issues or actionable_library_issue:
        actions.append(
            FixAction(
                kind="write_requirements",
                description=f"Generate requirements file for stack '{stack}'",
                risk="low",
                output_path=Path(requirements_output),
                details=(
                    "Generated from grouped root-cause findings"
                    if root_cause_requires_requirements
                    else "Generated from actionable raw issues"
                ),
            )
        )

    return actions


def _execute_action(
    action: FixAction,
    *,
    stack: str,
    python_executable: Optional[str],
) -> tuple[bool, Optional[Path], str]:
    """Execute a single fix action."""
    if action.kind == "create_venv":
        env_path = create_virtualenv(str(action.output_path or ".venv"))
        return env_path is not None, env_path, action.description

    if action.kind == "write_conda":
        output_path = generate_conda_env(stack=stack, output_file=str(action.output_path))
        return True, output_path, action.description

    if action.kind == "write_requirements":
        output_path = generate_requirements_txt(stack=stack, output_file=str(action.output_path))
        return True, output_path, action.description

    if action.kind == "install_requirements":
        if action.output_path is None:
            return False, None, "Missing requirements path for installation"
        success = install_requirements(
            str(action.output_path),
            use_conda=False,
            python_executable=python_executable,
        )
        return success, action.output_path, action.description

    return False, None, f"Unknown fix action: {action.kind}"


def auto_fix(
    use_conda: bool = False,
    create_venv: bool = False,
    stack: str = "trl-peft",
    *,
    apply: bool = False,
    yes: bool = False,
    dry_run: bool = False,
    verify: bool = True,
) -> FixResult:
    """Auto-fix environment issues based on diagnostics."""
    console.print(f"[bold blue]{icon_wrench()} Running Auto-Fix...[/bold blue]\n")

    config = load_config()
    if not apply and not dry_run and yes:
        apply = True

    if not apply and not dry_run and bool(config.get("fix", {}).get("auto_install", False)):
        apply = True

    issues = diagnose_env(full=False, show_header=False)
    doctor_findings = summarize_for_doctor(issues)
    selected_stack = recommend_stack_from_findings(stack, doctor_findings)
    actions = plan_fixes(
        issues,
        use_conda=use_conda,
        create_venv=create_venv,
        stack=selected_stack,
        doctor_findings=doctor_findings,
    )

    critical_issues = [
        issue for issue in issues if issue.severity == "critical" and "FAIL" in issue.status
    ]

    if not critical_issues and not actions:
        print_success("No critical issues found! Environment looks good.")
        return FixResult(
            success=True,
            verification_issues=issues,
            message="No fix actions required",
            reasons=[finding.problem for finding in doctor_findings],
            selected_stack=selected_stack,
        )

    if not actions:
        print_info("No automated fix actions are available for the current issues.")
        return FixResult(
            success=True,
            verification_issues=issues,
            message="No automated actions available",
            reasons=[finding.problem for finding in doctor_findings],
            selected_stack=selected_stack,
        )

    if doctor_findings:
        console.print("[bold]Why these actions are being planned:[/bold]")
        for finding in doctor_findings[:3]:
            console.print(f"[yellow]- {finding.problem}[/yellow]")
            console.print(f"  Likely cause: {finding.likely_cause}")

    console.print(f"[yellow]Planned {len(actions)} fix action(s)[/yellow]")
    console.print(f"[cyan]Selected stack: {selected_stack}[/cyan]")
    for action in actions:
        console.print(f"[cyan]- [{action.risk} risk] {action.description}[/cyan]")

    if dry_run:
        print_info("Dry run only. No changes were applied.")
        return FixResult(
            success=True,
            actions=actions,
            verification_issues=issues,
            message="Dry run complete",
            reasons=[finding.problem for finding in doctor_findings],
            selected_stack=selected_stack,
        )

    should_apply = apply
    if not apply and not yes:
        console.print()
        response = console.input("[bold yellow]Apply these fix actions now? (y/n): [/bold yellow]")
        should_apply = response.strip().lower() in {"y", "yes"}

    if not should_apply:
        print_info("No changes applied.")
        return FixResult(
            success=True,
            actions=actions,
            verification_issues=issues,
            message="User skipped apply",
            reasons=[finding.problem for finding in doctor_findings],
            selected_stack=selected_stack,
        )

    executed_actions: List[str] = []
    created_paths: List[Path] = []
    venv_python: Optional[str] = None
    success = True
    backup_candidates = [action.output_path for action in actions if action.output_path is not None]
    backup_dir = create_fix_backup([path for path in backup_candidates if path is not None])
    if backup_dir is not None:
        console.print(f"[cyan]Rollback backup created: {backup_dir}[/cyan]")

    for action in actions:
        ok, output_path, description = _execute_action(
            action,
            stack=selected_stack,
            python_executable=venv_python,
        )
        executed_actions.append(description)
        if output_path is not None:
            created_paths.append(output_path)
            if action.kind == "create_venv":
                venv_python = str(get_virtualenv_python(output_path))
        if not ok:
            success = False
            break

    requirements_path = next((path for path in created_paths if path.name.endswith(".txt")), None)
    if success and apply and requirements_path is not None and not use_conda:
        install_action = FixAction(
            kind="install_requirements",
            description=f"Install dependencies from {requirements_path}",
            risk="medium",
            output_path=requirements_path,
        )
        ok, _, description = _execute_action(
            install_action,
            stack=selected_stack,
            python_executable=venv_python,
        )
        executed_actions.append(description)
        success = success and ok

    verification_issues: List[DiagnosticIssue] = []
    verification_summary: Optional[FixVerificationSummary] = None
    if success and verify:
        verification_issues = diagnose_env(full=False, show_header=False)
        verification_summary = _build_verification_summary(issues, verification_issues)

    if success:
        print_success("Auto-fix completed.")
        if verification_summary is not None:
            console.print(
                "[bold]Verification summary:[/bold] "
                f"critical {verification_summary.before_critical} -> {verification_summary.after_critical}, "
                f"warnings {verification_summary.before_warning} -> {verification_summary.after_warning}, "
                f"confidence_delta={verification_summary.confidence_delta:+d}"
            )
            if verification_summary.resolved_critical:
                console.print(
                    f"[green]Resolved critical issues: {', '.join(verification_summary.resolved_critical)}[/green]"
                )
            if verification_summary.remaining_critical:
                print_warning(
                    "Verification found remaining critical issues. Review the updated diagnostics."
                )
                console.print(
                    f"[yellow]Remaining critical issues: {', '.join(verification_summary.remaining_critical)}[/yellow]"
                )
            else:
                print_success("Verification completed without critical issues.")
    else:
        print_error("Auto-fix stopped because one of the actions failed.")

    return FixResult(
        success=success,
        actions=actions,
        reasons=[finding.problem for finding in doctor_findings],
        selected_stack=selected_stack,
        executed_actions=executed_actions,
        verification_issues=verification_issues,
        verification_summary=verification_summary,
        created_paths=created_paths,
        message="Auto-fix completed" if success else "Auto-fix failed",
    )
