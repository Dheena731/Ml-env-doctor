"""Environment diagnostics for ML Environment Doctor."""

import importlib
import re
import subprocess
from typing import Dict, List, Optional, Tuple

try:
    import torch
except ImportError:
    torch = None  # type: ignore

from rich.table import Table

from .utils import (
    check_command_exists,
    console,
    format_size,
    get_home_config_dir,
    print_error,
    print_info,
    print_warning,
    run_command,
)


class DiagnosticIssue:
    """Represents a diagnostic issue."""

    def __init__(
        self,
        name: str,
        status: str,
        severity: str,
        fix: str,
        details: Optional[str] = None,
    ):
        self.name = name
        self.status = status
        self.severity = severity  # "critical", "warning", "info"
        self.fix = fix
        self.details = details

    def to_row(self) -> Tuple[str, str, str, str]:
        """Convert to table row."""
        status_icon = {
            "PASS": "✅",
            "FAIL": "❌",
            "WARN": "⚠️",
            "INFO": "ℹ️",
        }.get(self.status.split()[0], "❓")
        return (
            self.name,
            f"{status_icon} {self.status}",
            self.severity.upper(),
            self.fix,
        )


def check_cuda_driver() -> List[DiagnosticIssue]:
    """Check NVIDIA CUDA driver availability."""
    issues = []
    if not check_command_exists("nvidia-smi"):
        issues.append(
            DiagnosticIssue(
                name="NVIDIA GPU Driver",
                status="FAIL - nvidia-smi not found",
                severity="critical",
                fix="Install NVIDIA drivers: https://www.nvidia.com/Download/index.aspx",
            )
        )
        return issues

    try:
        result = run_command(["nvidia-smi"], timeout=10)
        if result.returncode != 0:
            issues.append(
                DiagnosticIssue(
                    name="NVIDIA GPU Driver",
                    status="FAIL - nvidia-smi error",
                    severity="critical",
                    fix="Install/reinstall NVIDIA drivers and reboot",
                )
            )
            return issues

        # Parse CUDA version from nvidia-smi
        output = result.stdout
        cuda_match = re.search(r"CUDA Version: (\d+\.\d+)", output)
        cuda_version = cuda_match.group(1) if cuda_match else "unknown"

        # Check for GPU
        if "Driver Version" in output:
            issues.append(
                DiagnosticIssue(
                    name="NVIDIA GPU Driver",
                    status=f"PASS - CUDA {cuda_version}",
                    severity="info",
                    fix="",
                    details=f"CUDA Version: {cuda_version}",
                )
            )
        else:
            issues.append(
                DiagnosticIssue(
                    name="NVIDIA GPU Driver",
                    status="FAIL - No GPU detected",
                    severity="critical",
                    fix="Check GPU installation and drivers",
                )
            )
    except Exception as e:
        issues.append(
            DiagnosticIssue(
                name="NVIDIA GPU Driver",
                status=f"FAIL - {str(e)}",
                severity="critical",
                fix="Install NVIDIA drivers and reboot",
            )
        )

    return issues


def check_pytorch_cuda() -> List[DiagnosticIssue]:
    """Check PyTorch CUDA availability."""
    issues = []
    if torch is None:
        issues.append(
            DiagnosticIssue(
                name="PyTorch Installation",
                status="FAIL - Not installed",
                severity="critical",
                fix="pip install torch --index-url https://download.pytorch.org/whl/cu124",
            )
        )
        return issues

    try:
        # Check PyTorch version
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()

        if cuda_available:
            cuda_version = torch.version.cuda or "unknown"
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "unknown"

            issues.append(
                DiagnosticIssue(
                    name="PyTorch CUDA",
                    status=f"PASS - CUDA {cuda_version} ({device_count} GPU(s))",
                    severity="info",
                    fix="",
                    details=f"PyTorch {torch_version}, Device: {device_name}",
                )
            )

            # Check PyTorch/CUDA version compatibility
            torch_major, torch_minor = map(int, torch_version.split(".")[:2])
            if torch_major < 2 or (torch_major == 2 and torch_minor < 4):
                issues.append(
                    DiagnosticIssue(
                        name="PyTorch Version",
                        status="WARN - Old version",
                        severity="warning",
                        fix=f"Upgrade: pip install torch>=2.4.0 --index-url https://download.pytorch.org/whl/cu124",
                        details=f"Current: {torch_version}, Recommended: >=2.4.0",
                    )
                )
        else:
            issues.append(
                DiagnosticIssue(
                    name="PyTorch CUDA",
                    status="FAIL - CUDA not available",
                    severity="critical",
                    fix="pip install torch --index-url https://download.pytorch.org/whl/cu124",
                    details=f"PyTorch {torch_version} installed but CUDA not available",
                )
            )
    except Exception as e:
        issues.append(
            DiagnosticIssue(
                name="PyTorch CUDA",
                status=f"FAIL - {str(e)}",
                severity="critical",
                fix="Reinstall PyTorch with CUDA support",
            )
        )

    return issues


def check_ml_libraries() -> List[DiagnosticIssue]:
    """Check ML library installations."""
    issues = []
    required_libs = {
        "transformers": ">=4.44.0",
        "peft": ">=0.12.0",
        "trl": ">=0.9.0",
        "datasets": ">=2.20.0",
        "accelerate": ">=1.0.0",
    }

    for lib_name, version_req in required_libs.items():
        try:
            module = importlib.import_module(lib_name)
            version = getattr(module, "__version__", "unknown")

            # Simple version check (basic)
            min_version = version_req.replace(">=", "")
            if version != "unknown":
                try:
                    from packaging import version as pkg_version

                    if pkg_version.parse(version) < pkg_version.parse(min_version):
                        issues.append(
                            DiagnosticIssue(
                                name=f"{lib_name}",
                                status=f"WARN - Old version ({version})",
                                severity="warning",
                                fix=f"pip install {lib_name}{version_req}",
                                details=f"Current: {version}, Required: {version_req}",
                            )
                        )
                    else:
                        issues.append(
                            DiagnosticIssue(
                                name=f"{lib_name}",
                                status=f"PASS - {version}",
                                severity="info",
                                fix="",
                            )
                        )
                except (ImportError, Exception):
                    # If packaging not available, just check if module exists
                    issues.append(
                        DiagnosticIssue(
                            name=f"{lib_name}",
                            status=f"PASS - Installed",
                            severity="info",
                            fix="",
                            details=f"Version: {version}",
                        )
                    )
            else:
                issues.append(
                    DiagnosticIssue(
                        name=f"{lib_name}",
                        status="PASS - Installed",
                        severity="info",
                        fix="",
                    )
                )
        except ImportError:
            issues.append(
                DiagnosticIssue(
                    name=f"{lib_name}",
                    status="FAIL - Not installed",
                    severity="critical",
                    fix=f"pip install {lib_name}{version_req}",
                )
            )

    return issues


def check_gpu_memory() -> List[DiagnosticIssue]:
    """Check GPU memory availability."""
    issues = []
    if torch is None or not torch.cuda.is_available():
        return issues

    try:
        free_mem, total_mem = torch.cuda.mem_get_info(0)
        free_gb = free_mem / (1024**3)
        total_gb = total_mem / (1024**3)
        used_gb = total_gb - free_gb

        if free_gb < 8:
            issues.append(
                DiagnosticIssue(
                    name="GPU Memory",
                    status=f"WARN - Low memory ({free_gb:.1f}GB free)",
                    severity="warning",
                    fix="Close other processes or use smaller models",
                    details=f"Free: {free_gb:.1f}GB / Total: {total_gb:.1f}GB",
                )
            )
        else:
            issues.append(
                DiagnosticIssue(
                    name="GPU Memory",
                    status=f"PASS - {free_gb:.1f}GB free",
                    severity="info",
                    fix="",
                    details=f"Free: {free_gb:.1f}GB / Total: {total_gb:.1f}GB",
                )
            )
    except Exception as e:
        issues.append(
            DiagnosticIssue(
                name="GPU Memory",
                status=f"FAIL - {str(e)}",
                severity="warning",
                fix="Check GPU access",
            )
        )

    return issues


def check_disk_space() -> List[DiagnosticIssue]:
    """Check disk space for HF cache."""
    issues = []
    try:
        import shutil

        cache_dir = get_home_config_dir().parent / ".cache" / "huggingface"
        if cache_dir.exists():
            stat = shutil.disk_usage(cache_dir.parent)
            free_gb = stat.free / (1024**3)
            if free_gb < 50:
                issues.append(
                    DiagnosticIssue(
                        name="Disk Space",
                        status=f"WARN - Low space ({free_gb:.1f}GB free)",
                        severity="warning",
                        fix="Free up disk space (HF cache needs ~50GB)",
                        details=f"Free: {format_size(stat.free)}",
                    )
                )
            else:
                issues.append(
                    DiagnosticIssue(
                        name="Disk Space",
                        status=f"PASS - {free_gb:.1f}GB free",
                        severity="info",
                        fix="",
                        details=f"Free: {format_size(stat.free)}",
                    )
                )
        else:
            issues.append(
                DiagnosticIssue(
                    name="Disk Space",
                    status="INFO - Cache dir not found",
                    severity="info",
                    fix="",
                )
            )
    except Exception as e:
        issues.append(
            DiagnosticIssue(
                name="Disk Space",
                status=f"WARN - {str(e)}",
                severity="warning",
                fix="Check disk space manually",
            )
        )

    return issues


def check_docker_gpu() -> List[DiagnosticIssue]:
    """Check Docker GPU support."""
    issues = []
    if not check_command_exists("docker"):
        issues.append(
            DiagnosticIssue(
                name="Docker GPU Support",
                status="INFO - Docker not installed",
                severity="info",
                fix="Install Docker for GPU containerization",
            )
        )
        return issues

    try:
        result = run_command(
            [
                "docker",
                "run",
                "--rm",
                "--gpus",
                "all",
                "nvidia/cuda:12.4.0-base-ubuntu22.04",
                "nvidia-smi",
            ],
            timeout=30,
        )
        if result.returncode == 0:
            issues.append(
                DiagnosticIssue(
                    name="Docker GPU Support",
                    status="PASS - nvidia-docker working",
                    severity="info",
                    fix="",
                )
            )
        else:
            issues.append(
                DiagnosticIssue(
                    name="Docker GPU Support",
                    status="FAIL - GPU not accessible in Docker",
                    severity="warning",
                    fix="Install nvidia-container-toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html",
                )
            )
    except Exception:
        issues.append(
            DiagnosticIssue(
                name="Docker GPU Support",
                status="INFO - Docker GPU test skipped",
                severity="info",
                fix="Test manually: docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi",
            )
        )

    return issues


def check_internet_connectivity() -> List[DiagnosticIssue]:
    """Check internet connectivity for HF Hub."""
    issues = []
    try:
        import urllib.request

        urllib.request.urlopen("https://huggingface.co", timeout=5)
        issues.append(
            DiagnosticIssue(
                name="Internet Connectivity",
                status="PASS - HF Hub accessible",
                severity="info",
                fix="",
            )
        )
    except Exception:
        issues.append(
            DiagnosticIssue(
                name="Internet Connectivity",
                status="WARN - Cannot reach HF Hub",
                severity="warning",
                fix="Check internet connection and firewall settings",
            )
        )

    return issues


def diagnose_env(full: bool = False) -> List[DiagnosticIssue]:
    """Run all diagnostic checks."""
    all_issues: List[DiagnosticIssue] = []

    console.print("[bold blue]🔍 Running ML Environment Diagnostics...[/bold blue]\n")

    # Core checks (always run)
    all_issues.extend(check_cuda_driver())
    all_issues.extend(check_pytorch_cuda())
    all_issues.extend(check_ml_libraries())

    # Extended checks (if --full)
    if full:
        all_issues.extend(check_gpu_memory())
        all_issues.extend(check_disk_space())
        all_issues.extend(check_docker_gpu())
        all_issues.extend(check_internet_connectivity())

    return all_issues


def print_diagnostic_table(issues: List[DiagnosticIssue]) -> None:
    """Print diagnostic results as a Rich table."""
    table = Table(
        title="ML Environment Doctor - Diagnostic Results",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Issue", style="cyan", no_wrap=False)
    table.add_column("Status", style="bold")
    table.add_column("Severity", style="yellow")
    table.add_column("Fix", style="green", no_wrap=False)

    for issue in issues:
        table.add_row(*issue.to_row())

    console.print()
    console.print(table)

    # Summary
    critical_count = sum(1 for i in issues if i.severity == "critical" and "FAIL" in i.status)
    warning_count = sum(
        1 for i in issues if i.severity == "warning" and ("WARN" in i.status or "FAIL" in i.status)
    )
    pass_count = sum(1 for i in issues if "PASS" in i.status)

    console.print()
    console.print(f"[green]✅ Passed: {pass_count}[/green]")
    if warning_count > 0:
        console.print(f"[yellow]⚠️  Warnings: {warning_count}[/yellow]")
    if critical_count > 0:
        console.print(f"[red]❌ Critical Issues: {critical_count}[/red]")

    if critical_count == 0 and warning_count == 0:
        console.print(
            "\n[bold green]🎉 Your ML environment looks ready for fine-tuning![/bold green]"
        )
    elif critical_count > 0:
        console.print("\n[bold red]⚠️  Please fix critical issues before proceeding.[/bold red]")
    else:
        console.print(
            "\n[bold yellow]💡 Consider addressing warnings for optimal performance.[/bold yellow]"
        )
