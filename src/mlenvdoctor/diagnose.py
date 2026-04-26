"""Environment diagnostics for ML Environment Doctor."""

from __future__ import annotations

import importlib
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from rich.table import Table

from .constants import (
    MIN_ACCELERATE_VERSION,
    MIN_DATASETS_VERSION,
    MIN_DISK_SPACE_GB,
    MIN_GPU_MEMORY_GB,
    MIN_PEFT_VERSION,
    MIN_PYTHON_VERSION,
    MIN_PYTORCH_VERSION,
    MIN_TRANSFORMERS_VERSION,
    MIN_TRL_VERSION,
)
from .icons import icon_check, icon_cross, icon_info, icon_search, icon_warning
from .utils import check_command_exists, console, format_size, get_home_config_dir, run_command

try:
    from packaging import version as pkg_version
except ImportError:  # pragma: no cover - packaging is a required dependency
    pkg_version = None  # type: ignore[assignment]

try:
    import torch
except ImportError:  # pragma: no cover - depends on user environment
    torch = None  # type: ignore


@dataclass
class DiagnosticIssue:
    """Represents a diagnostic outcome."""

    name: str
    status: str
    severity: str
    fix: str
    details: Optional[str] = None
    check_id: str = ""
    category: str = "general"
    recommendation: str = ""
    likely_cause: str = ""
    verify_steps: List[str] = field(default_factory=list)
    confidence: str = "medium"
    evidence: List[str] = field(default_factory=list)
    mismatch_code: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> Tuple[str, str, str, str]:
        """Convert to table row."""
        status_icon_map = {
            "PASS": icon_check(),
            "FAIL": icon_cross(),
            "WARN": icon_warning(),
            "INFO": icon_info(),
        }
        status_icon = status_icon_map.get(self.status.split()[0], "?")
        return (
            self.name,
            f"{status_icon} {self.status}",
            self.severity.upper(),
            self.fix,
        )


@dataclass
class DoctorFinding:
    """Prioritized summary item for the `doctor` command."""

    problem: str
    severity: str
    confidence: str
    likely_cause: str
    best_fix: str
    verify_steps: List[str]
    evidence: List[str]
    check_id: str
    category: str
    linked_checks: List[str] = field(default_factory=list)


def _find_issue(
    issues: List["DiagnosticIssue"], check_id: str, status_prefixes: Tuple[str, ...]
) -> Optional["DiagnosticIssue"]:
    """Return the first issue matching a selector."""
    for issue in issues:
        if issue.check_id == check_id and issue.status.startswith(status_prefixes):
            return issue
    return None


def make_issue(
    *,
    name: str,
    status: str,
    severity: str,
    fix: str = "",
    details: Optional[str] = None,
    check_id: str,
    category: str,
    recommendation: str = "",
    likely_cause: str = "",
    verify_steps: Optional[List[str]] = None,
    confidence: str = "medium",
    evidence: Optional[List[str]] = None,
    mismatch_code: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> DiagnosticIssue:
    """Create a normalized diagnostic issue."""
    return DiagnosticIssue(
        name=name,
        status=status,
        severity=severity,
        fix=fix,
        details=details,
        check_id=check_id,
        category=category,
        recommendation=recommendation or fix,
        likely_cause=likely_cause,
        verify_steps=verify_steps or [],
        confidence=confidence,
        evidence=evidence or [],
        mismatch_code=mismatch_code,
        metadata=metadata or {},
    )


def _is_version_at_least(current: str, minimum: str) -> Optional[bool]:
    """Compare versions when the packaging helper is available."""
    if pkg_version is None or current == "unknown":
        return None
    return pkg_version.parse(current) >= pkg_version.parse(minimum)


def _summarize_exception(exc: Exception) -> str:
    """Return a short exception summary for diagnostics."""
    return f"{type(exc).__name__}: {exc}"


COMPATIBILITY_BASELINES: Dict[str, Dict[str, str]] = {
    "tensorflow_windows": {"max_recommended_python": "3.11"},
    "tensorflow_general": {"max_recommended_python": "3.12"},
    "pytorch_cuda": {"recommended_cuda": "12.4"},
}


def _default_verify_steps(issue: DiagnosticIssue) -> List[str]:
    """Return fallback verification steps for a diagnostic issue."""
    verify_by_check_id = {
        "accelerator_backend": ['python -c "import torch; print(torch.cuda.is_available())"'],
        "python_runtime": ["python --version"],
        "cuda_driver": ["nvidia-smi"],
        "pytorch_installation": ['python -c "import torch; print(torch.__version__)"'],
        "pytorch_version": ['python -c "import torch; print(torch.__version__)"'],
        "pytorch_cuda": ['python -c "import torch; print(torch.cuda.is_available())"'],
        "pytorch_cuda_execution": [
            "python -c \"import torch; x=torch.tensor([1.0], device='cuda:0'); print(x.item())\""
        ],
        "tensorflow_runtime": [
            "python -c \"import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))\""
        ],
        "tensorflow_execution": [
            'python -c "import tensorflow as tf; print(tf.matmul(tf.constant([[1.0]]), tf.constant([[2.0]])).numpy())"'
        ],
        "keras_version": ['python -c "import keras; print(keras.__version__)"'],
        "jax_runtime": [
            'python -c "import jax; print(jax.default_backend()); print(jax.devices())"'
        ],
        "jax_execution": ['python -c "import jax.numpy as jnp; print(jnp.array([1, 2, 3]).sum())"'],
        "flax_runtime": ['python -c "import flax; print(flax.__version__)"'],
        "gpu_memory": ["nvidia-smi"],
        "disk_space": ["df -h"],
        "docker_gpu": ["docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi"],
        "internet_connectivity": [
            "python -c \"import urllib.request; urllib.request.urlopen('https://huggingface.co', timeout=5)\""
        ],
    }
    return verify_by_check_id.get(issue.check_id, ["mlenvdoctor diagnose"])


def _default_likely_cause(issue: DiagnosticIssue) -> str:
    """Infer a likely cause when one is not provided explicitly."""
    cause_by_check_id = {
        "accelerator_backend": "The detected accelerator path does not match the expected runtime for this machine.",
        "python_runtime": "The active interpreter is older than the minimum version supported by this package.",
        "cuda_driver": "The NVIDIA driver stack is missing, misconfigured, or not visible from the current shell.",
        "pytorch_installation": "PyTorch is not installed in the active environment.",
        "pytorch_version": "PyTorch is installed but below the recommended baseline for this tool's target workflows.",
        "pytorch_cuda": "PyTorch is installed, but the active build cannot see or use CUDA on this machine.",
        "pytorch_cuda_execution": "CUDA is reported as available, but a real tensor operation failed at runtime.",
        "tensorflow_runtime": "TensorFlow is missing or its runtime is not correctly configured for the available hardware.",
        "tensorflow_execution": "TensorFlow imports, but a basic tensor operation failed inside the active runtime.",
        "keras_version": "Keras is missing or older than the recommended version for current TensorFlow workflows.",
        "jax_runtime": "JAX is missing or is falling back to CPU instead of the expected accelerator backend.",
        "jax_execution": "JAX imports, but a simple array operation failed in the active runtime.",
        "flax_runtime": "Flax is not installed in the active environment.",
        "gpu_memory": "Other GPU workloads are already using too much memory for the target ML workflow.",
        "disk_space": "The cache or working drive does not have enough free space for model downloads and artifacts.",
        "docker_gpu": "Docker can run, but GPU passthrough is not configured correctly inside containers.",
        "internet_connectivity": "The current machine cannot reach Hugging Face Hub from this network or environment.",
    }
    return cause_by_check_id.get(
        issue.check_id,
        issue.details or "The tool detected a problem but could not infer a more specific cause.",
    )


def _platform_hint() -> str:
    """Return a short platform label for user-facing guidance."""
    if _is_wsl_environment():
        return "wsl2"
    if sys.platform == "win32":
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "linux"


def _is_wsl_environment() -> bool:
    """Return whether diagnostics are running inside WSL."""
    if sys.platform != "linux":
        return False

    release_path = Path("/proc/sys/kernel/osrelease")
    version_path = Path("/proc/version")
    for path in (release_path, version_path):
        try:
            value = path.read_text(encoding="utf-8", errors="ignore").lower()
            if "microsoft" in value or "wsl" in value:
                return True
        except OSError:
            continue
    return False


def _issue_matches(issue: DiagnosticIssue, check_id: str, status_prefixes: Tuple[str, ...]) -> bool:
    """Return whether an issue matches a rule selector."""
    return issue.check_id == check_id and issue.status.startswith(status_prefixes)


def _has_issue(
    issues: List[DiagnosticIssue], check_id: str, status_prefixes: Tuple[str, ...]
) -> bool:
    """Check if the issue list contains a matching issue."""
    return any(_issue_matches(issue, check_id, status_prefixes) for issue in issues)


def _collect_evidence(issues: List[DiagnosticIssue], check_ids: List[str]) -> List[str]:
    """Collect the best evidence lines for linked checks."""
    evidence: List[str] = []
    for issue in issues:
        if issue.check_id in check_ids:
            if issue.evidence:
                evidence.extend(issue.evidence[:2])
            elif issue.details:
                evidence.append(issue.details)
            else:
                evidence.append(issue.status)
    return evidence[:4]


def _root_cause_findings(issues: List[DiagnosticIssue]) -> List[DoctorFinding]:
    """Infer higher-level doctor findings from detailed checks."""
    findings: List[DoctorFinding] = []

    if _has_issue(issues, "cuda_driver", ("FAIL",)):
        platform_hint = _platform_hint()
        likely_cause = (
            "The NVIDIA driver is missing, broken, or not visible from the current shell."
        )
        best_fix = "Install or repair the NVIDIA driver, then re-run `nvidia-smi` before retrying framework installs."
        if platform_hint == "wsl2":
            likely_cause = (
                "WSL cannot currently see the host NVIDIA stack. This usually means GPU passthrough is not enabled "
                "or host/WSL NVIDIA components are out of sync."
            )
            best_fix = (
                "Verify GPU support on the Windows host first, then update WSL and NVIDIA WSL drivers/tooling "
                "before retrying inside WSL."
            )
        elif platform_hint == "macos":
            likely_cause = "NVIDIA CUDA tooling is not available in this runtime. On macOS, CUDA-first guidance is usually not applicable."
            best_fix = "Use Apple-supported acceleration paths such as MPS when available, or continue with CPU-focused workflows."
        findings.append(
            DoctorFinding(
                problem="GPU driver stack is not usable",
                severity="critical",
                confidence="high",
                likely_cause=likely_cause,
                best_fix=best_fix,
                verify_steps=["nvidia-smi", "mlenvdoctor doctor"],
                evidence=_collect_evidence(issues, ["cuda_driver"]),
                check_id="root_gpu_driver",
                category="gpu",
                linked_checks=["cuda_driver"],
            )
        )

    backend_issue = _find_issue(issues, "accelerator_backend", ("WARN",))
    if backend_issue is not None:
        findings.append(
            DoctorFinding(
                problem="Accelerator backend path needs attention",
                severity="warning",
                confidence=backend_issue.confidence,
                likely_cause=backend_issue.likely_cause
                or "The current runtime backend is different from the expected accelerator path.",
                best_fix=backend_issue.recommendation
                or backend_issue.fix
                or "Review runtime backend configuration.",
                verify_steps=backend_issue.verify_steps or _default_verify_steps(backend_issue),
                evidence=backend_issue.evidence
                or ([backend_issue.details] if backend_issue.details else []),
                check_id="root_accelerator_backend",
                category="platform",
                linked_checks=["accelerator_backend"],
            )
        )

    if _has_issue(issues, "pytorch_installation", ("FAIL",)):
        findings.append(
            DoctorFinding(
                problem="PyTorch is missing from the active environment",
                severity="critical",
                confidence="high",
                likely_cause="The current Python environment does not have PyTorch installed.",
                best_fix="Install a PyTorch build that matches your target hardware, then rerun `mlenvdoctor doctor`.",
                verify_steps=[
                    'python -c "import torch; print(torch.__version__)"',
                    'python -c "import torch; print(torch.cuda.is_available())"',
                ],
                evidence=_collect_evidence(issues, ["pytorch_installation"]),
                check_id="root_pytorch_missing",
                category="pytorch",
                linked_checks=["pytorch_installation"],
            )
        )

    pytorch_cuda_issue = _find_issue(issues, "pytorch_cuda", ("FAIL",))
    if (
        _has_issue(issues, "cuda_driver", ("PASS",))
        and pytorch_cuda_issue is not None
        and not _has_issue(issues, "pytorch_installation", ("FAIL",))
    ):
        has_build_metadata = "torch_cuda_build" in pytorch_cuda_issue.metadata
        torch_cuda_build = pytorch_cuda_issue.metadata.get("torch_cuda_build")
        if has_build_metadata and torch_cuda_build is None:
            problem = "PyTorch is installed as a CPU-only build on a GPU-capable machine"
            likely_cause = (
                "The machine exposes an NVIDIA driver, but the installed PyTorch wheel was built "
                "without CUDA support."
            )
            best_fix = (
                "Install the CUDA-enabled PyTorch wheel for this environment, then verify that "
                "`torch.version.cuda` is no longer empty."
            )
            check_id = "root_pytorch_cpu_only_build"
        else:
            problem = "PyTorch cannot use the available GPU"
            likely_cause = (
                "The machine exposes an NVIDIA driver, but the installed PyTorch build is mismatched "
                "with the local CUDA or runtime setup."
            )
            best_fix = (
                "Reinstall the correct CUDA-enabled PyTorch wheel for this environment, then verify "
                "CUDA visibility in Python."
            )
            check_id = "root_pytorch_cuda_mismatch"

        findings.append(
            DoctorFinding(
                problem=problem,
                severity="critical",
                confidence="high",
                likely_cause=likely_cause,
                best_fix=best_fix,
                verify_steps=[
                    'python -c "import torch; print(torch.__version__); print(torch.version.cuda)"',
                    'python -c "import torch; print(torch.cuda.is_available())"',
                ],
                evidence=_collect_evidence(
                    issues, ["cuda_driver", "pytorch_cuda", "pytorch_version"]
                ),
                check_id=check_id,
                category="pytorch",
                linked_checks=["cuda_driver", "pytorch_cuda", "pytorch_version"],
            )
        )

    if _has_issue(issues, "tensorflow_runtime", ("WARN",)):
        platform_hint = _platform_hint()
        likely_cause = "TensorFlow is installed, but the active runtime does not include working GPU support for this machine."
        best_fix = "Install or repair the TensorFlow GPU-supported runtime appropriate for this platform, then recheck visible GPU devices."
        if platform_hint == "windows":
            likely_cause = (
                "TensorFlow is installed on Windows, but the active setup is not using a supported GPU path. "
                "Native Windows TensorFlow GPU workflows are limited compared with Linux/WSL2 paths."
            )
            best_fix = (
                "Prefer a supported Windows path such as WSL2 for CUDA-based TensorFlow workflows, "
                "or use a Windows-appropriate backend before rechecking GPU visibility."
            )
        elif platform_hint == "macos":
            likely_cause = (
                "TensorFlow is installed on macOS, where CUDA guidance is not applicable and accelerator support "
                "depends on Apple-specific runtime paths."
            )
            best_fix = (
                "Use the macOS-supported TensorFlow path for Apple hardware and verify available devices again "
                "instead of following CUDA-specific instructions."
            )
        findings.append(
            DoctorFinding(
                problem="TensorFlow is falling back away from the expected GPU path",
                severity="warning",
                confidence="medium",
                likely_cause=likely_cause,
                best_fix=best_fix,
                verify_steps=[
                    "python -c \"import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))\"",
                    "mlenvdoctor diagnose",
                ],
                evidence=_collect_evidence(issues, ["tensorflow_runtime", "tensorflow_execution"]),
                check_id="root_tensorflow_gpu_path",
                category="tensorflow",
                linked_checks=["tensorflow_runtime", "tensorflow_execution"],
            )
        )

    if _has_issue(issues, "jax_runtime", ("WARN",)):
        platform_hint = _platform_hint()
        likely_cause = "JAX is installed, but the active environment lacks a working accelerator-enabled backend for this machine."
        best_fix = "Install the correct JAX backend for your platform, then verify `jax.default_backend()` again."
        if platform_hint == "windows":
            likely_cause = (
                "JAX is installed on Windows, but the active environment is using the CPU backend. "
                "Accelerator-enabled JAX support is more constrained than on Linux."
            )
            best_fix = (
                "Prefer a supported accelerator path such as Linux/WSL2 for GPU-backed JAX workflows, "
                "or accept CPU-only execution for this environment."
            )
        elif platform_hint == "macos":
            likely_cause = (
                "JAX is installed on macOS, but the active backend is still CPU-only. "
                "Accelerator availability depends on the local Apple-supported runtime path."
            )
            best_fix = (
                "Verify the macOS-supported JAX backend for this hardware, or continue with CPU-only execution "
                "if accelerator support is not available."
            )
        findings.append(
            DoctorFinding(
                problem="JAX is using the CPU backend unexpectedly",
                severity="warning",
                confidence="medium",
                likely_cause=likely_cause,
                best_fix=best_fix,
                verify_steps=[
                    'python -c "import jax; print(jax.default_backend()); print(jax.devices())"',
                    "mlenvdoctor diagnose",
                ],
                evidence=_collect_evidence(issues, ["jax_runtime", "jax_execution"]),
                check_id="root_jax_backend",
                category="jax",
                linked_checks=["jax_runtime", "jax_execution"],
            )
        )

    if _has_issue(issues, "docker_gpu", ("FAIL", "WARN")) and _has_issue(
        issues, "cuda_driver", ("PASS",)
    ):
        findings.append(
            DoctorFinding(
                problem="Docker GPU passthrough is misconfigured",
                severity="warning",
                confidence="high",
                likely_cause="The host can see the NVIDIA stack, but container GPU access is not configured correctly.",
                best_fix="Repair the NVIDIA Container Toolkit or Docker GPU runtime configuration before relying on GPU containers.",
                verify_steps=[
                    "docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi",
                    "mlenvdoctor diagnose --full",
                ],
                evidence=_collect_evidence(issues, ["cuda_driver", "docker_gpu"]),
                check_id="root_docker_gpu_runtime",
                category="docker",
                linked_checks=["cuda_driver", "docker_gpu"],
            )
        )

    if _has_issue(issues, "gpu_memory", ("WARN",)):
        findings.append(
            DoctorFinding(
                problem="GPU memory pressure is too high for the current workload",
                severity="warning",
                confidence="high",
                likely_cause="Other processes or previous allocations are using too much VRAM for the intended ML task.",
                best_fix="Free GPU memory, reduce workload size, or switch to a smaller model or batch size before retrying.",
                verify_steps=["nvidia-smi", "mlenvdoctor diagnose --full"],
                evidence=_collect_evidence(issues, ["gpu_memory"]),
                check_id="root_gpu_memory_pressure",
                category="gpu",
                linked_checks=["gpu_memory"],
            )
        )

    if _has_issue(issues, "disk_space", ("WARN",)):
        findings.append(
            DoctorFinding(
                problem="Local disk space is too low for reliable ML workflows",
                severity="warning",
                confidence="high",
                likely_cause="The cache or working volume does not have enough room for downloads, model files, or artifacts.",
                best_fix="Free disk space or move caches to a larger drive before retrying downloads or training jobs.",
                verify_steps=["df -h", "mlenvdoctor diagnose --full"],
                evidence=_collect_evidence(issues, ["disk_space"]),
                check_id="root_disk_space",
                category="system",
                linked_checks=["disk_space"],
            )
        )

    dependency_issues = [
        issue
        for issue in issues
        if issue.category == "dependencies" and issue.status.startswith(("FAIL", "WARN"))
    ]
    if len(dependency_issues) >= 2:
        failing_libs = [issue.name for issue in dependency_issues]
        has_missing = any(issue.status.startswith("FAIL") for issue in dependency_issues)
        findings.append(
            DoctorFinding(
                problem="The core ML training dependency stack is incomplete or outdated",
                severity="critical" if has_missing else "warning",
                confidence="high" if len(dependency_issues) >= 3 else "medium",
                likely_cause=(
                    "The active environment is missing or lagging on multiple libraries required for "
                    "modern training and fine-tuning workflows."
                ),
                best_fix=(
                    "Install or refresh the recommended training stack together instead of fixing "
                    "packages one by one."
                ),
                verify_steps=[
                    "mlenvdoctor stack llm-training",
                    "mlenvdoctor diagnose",
                ],
                evidence=failing_libs[:4],
                check_id="root_ml_stack_dependencies",
                category="dependencies",
                linked_checks=[issue.check_id for issue in dependency_issues],
            )
        )

    return findings


def _compatibility_matrix_issues(issues: List[DiagnosticIssue]) -> List[DiagnosticIssue]:
    """Infer compatibility mismatches using cross-check rules."""
    inferred: List[DiagnosticIssue] = []
    platform_hint = _platform_hint()
    py_issue = next((issue for issue in issues if issue.check_id == "python_runtime"), None)
    py_version = ""
    if py_issue is not None:
        py_version = py_issue.metadata.get("current_version", "")
        if not py_version and "PASS -" in py_issue.status:
            py_version = py_issue.status.replace("PASS -", "").strip()

    tf_issue = next((issue for issue in issues if issue.check_id == "tensorflow_runtime"), None)
    if tf_issue is not None and py_version and py_version != "unknown":
        if platform_hint == "windows":
            max_py = COMPATIBILITY_BASELINES["tensorflow_windows"]["max_recommended_python"]
            py_ok = _is_version_at_least(max_py, py_version)
            if py_ok is False:
                inferred.append(
                    make_issue(
                        name="Compatibility Matrix",
                        status=(
                            f"WARN - Python {py_version} may be ahead of the most stable "
                            f"TensorFlow path on Windows"
                        ),
                        severity="warning",
                        fix=(
                            f"Use Python <= {max_py} for the most stable Windows TensorFlow setup, "
                            "or use WSL2 for CUDA-based TensorFlow workflows."
                        ),
                        check_id="compatibility_matrix",
                        category="compatibility",
                        details="Detected TensorFlow on Windows with a newer Python runtime.",
                        confidence="medium",
                        mismatch_code="PYTHON_TF_COMPAT_WARNING",
                        evidence=[f"platform={platform_hint}", f"python={py_version}"],
                        metadata={"platform": platform_hint, "python_version": py_version},
                    )
                )

    cuda_issue = next((issue for issue in issues if issue.check_id == "cuda_driver"), None)
    torch_cuda_issue = next((issue for issue in issues if issue.check_id == "pytorch_cuda"), None)
    if (
        cuda_issue is not None
        and torch_cuda_issue is not None
        and cuda_issue.status.startswith("PASS")
        and torch_cuda_issue.status.startswith("FAIL")
    ):
        torch_cuda_build = torch_cuda_issue.metadata.get("torch_cuda_build")
        mismatch_code = "PT_CUDA_RUNTIME_MISMATCH"
        summary = "PyTorch CUDA runtime mismatch detected"
        if torch_cuda_build is None:
            mismatch_code = "PT_CPU_WHEEL_ON_GPU"
            summary = "PyTorch CPU-only wheel detected on a GPU-capable machine"
        inferred.append(
            make_issue(
                name="Compatibility Matrix",
                status=f"WARN - {summary}",
                severity="warning",
                fix=(
                    "Install a CUDA-enabled PyTorch build matching this environment and verify "
                    "`torch.version.cuda` and `torch.cuda.is_available()`."
                ),
                check_id="compatibility_matrix",
                category="compatibility",
                details=("NVIDIA driver is visible, but PyTorch still reports CUDA unavailable."),
                confidence="high",
                mismatch_code=mismatch_code,
                evidence=[
                    cuda_issue.status,
                    torch_cuda_issue.status,
                    f"torch_cuda_build={torch_cuda_build or 'None'}",
                ],
                metadata={"torch_cuda_build": torch_cuda_build},
            )
        )

    return inferred


def check_accelerator_backend() -> List[DiagnosticIssue]:
    """Detect the active accelerator backend and highlight platform mismatches."""
    platform_hint = _platform_hint()
    nvidia_visible = check_command_exists("nvidia-smi")

    torch_module = torch
    if torch_module is None:
        try:
            torch_module = importlib.import_module("torch")
        except Exception:
            torch_module = None

    if torch_module is None:
        details = "PyTorch not installed; backend detection is limited to host tooling."
        if platform_hint == "wsl2":
            details = "WSL environment detected without PyTorch installed. Install PyTorch to validate CUDA passthrough."
        status = "INFO - Backend detection limited (PyTorch not installed)"
        return [
            make_issue(
                name="Accelerator Backend",
                status=status,
                severity="info",
                fix="Install PyTorch and rerun diagnostics to detect runtime backend",
                check_id="accelerator_backend",
                category="platform",
                details=details,
                confidence="medium",
                evidence=[f"platform={platform_hint}", f"nvidia_visible={nvidia_visible}"],
                metadata={"platform": platform_hint, "nvidia_visible": nvidia_visible},
            )
        ]

    cuda_available = False
    mps_available = False
    try:
        cuda_available = bool(torch_module.cuda.is_available())
    except Exception:
        cuda_available = False
    try:
        mps_available = bool(
            getattr(torch_module.backends, "mps", None) and torch_module.backends.mps.is_available()
        )
    except Exception:
        mps_available = False

    backend = "cpu"
    if cuda_available:
        backend = "cuda"
    elif mps_available:
        backend = "mps"

    status = f"PASS - backend={backend}"
    severity = "info"
    fix = ""
    likely_cause = ""
    recommendation = ""
    verify_steps = ['python -c "import torch; print(torch.cuda.is_available())"']

    if platform_hint == "wsl2" and nvidia_visible and backend != "cuda":
        status = f"WARN - backend={backend} in WSL2 with NVIDIA host visibility"
        severity = "warning"
        likely_cause = (
            "WSL can see NVIDIA tooling, but the active PyTorch runtime is not using CUDA."
        )
        fix = (
            "Install a CUDA-enabled PyTorch build inside WSL and confirm host/WSL driver alignment."
        )
        recommendation = fix
        verify_steps = [
            "nvidia-smi",
            'python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"',
        ]
    elif platform_hint == "macos" and backend == "cpu":
        status = "WARN - backend=cpu on macOS"
        severity = "warning"
        likely_cause = (
            "The current runtime is using CPU only on macOS; MPS may be unavailable or not enabled."
        )
        fix = "Use an Apple Silicon-compatible PyTorch build and verify MPS availability if accelerator use is expected."
        recommendation = fix
        verify_steps = [
            "python -c \"import torch; print(hasattr(torch.backends, 'mps')); print(getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())\"",
        ]
    elif platform_hint == "windows" and nvidia_visible and backend == "cpu":
        status = "WARN - backend=cpu on Windows with NVIDIA tooling"
        severity = "warning"
        likely_cause = "PyTorch is running CPU-only despite NVIDIA tooling being present."
        fix = (
            "Use a CUDA-enabled PyTorch build on a supported path (often WSL2 for CUDA workflows)."
        )
        recommendation = fix

    return [
        make_issue(
            name="Accelerator Backend",
            status=status,
            severity=severity,
            fix=fix,
            check_id="accelerator_backend",
            category="platform",
            details=f"Platform: {platform_hint}, nvidia-smi visible: {nvidia_visible}",
            recommendation=recommendation or fix,
            likely_cause=likely_cause,
            verify_steps=verify_steps,
            confidence="high" if severity == "info" else "medium",
            evidence=[
                f"platform={platform_hint}",
                f"backend={backend}",
                f"nvidia_visible={nvidia_visible}",
            ],
            metadata={
                "platform": platform_hint,
                "backend": backend,
                "nvidia_visible": nvidia_visible,
            },
        )
    ]


def summarize_for_doctor(issues: List["DiagnosticIssue"]) -> List[DoctorFinding]:
    """Convert detailed issues into prioritized doctor findings."""
    findings = _root_cause_findings(issues)
    if findings:
        findings.sort(
            key=lambda finding: (
                0 if finding.severity == "critical" else 1,
                finding.category,
                finding.problem,
            )
        )
        return findings

    actionable_issues = [
        issue
        for issue in issues
        if issue.status.startswith(("FAIL", "WARN")) and issue.severity in {"critical", "warning"}
    ]
    actionable_issues.sort(
        key=lambda issue: (
            0 if issue.severity == "critical" else 1,
            issue.category,
            issue.name,
        )
    )

    fallback_findings: List[DoctorFinding] = []
    for issue in actionable_issues:
        fallback_findings.append(
            DoctorFinding(
                problem=issue.name,
                severity=issue.severity,
                confidence=issue.confidence,
                likely_cause=issue.likely_cause or _default_likely_cause(issue),
                best_fix=issue.recommendation
                or issue.fix
                or "Run `mlenvdoctor diagnose` for more detail.",
                verify_steps=issue.verify_steps or _default_verify_steps(issue),
                evidence=issue.evidence or ([issue.details] if issue.details else []),
                check_id=issue.check_id,
                category=issue.category,
                linked_checks=[issue.check_id] if issue.check_id else [],
            )
        )
    return fallback_findings


def get_fix_commands(issues: List["DiagnosticIssue"]) -> List[Dict[str, str]]:
    """Return copy-paste fix commands for actionable issues."""
    fixes: List[Dict[str, str]] = []
    for issue in issues:
        if issue.fix and issue.severity != "info" and issue.status.startswith(("FAIL", "WARN")):
            fixes.append(
                {
                    "issue": issue.name,
                    "severity": issue.severity,
                    "status": issue.status,
                    "command": issue.fix,
                    "check_id": issue.check_id,
                    "category": issue.category,
                }
            )
    return fixes


def check_python_runtime() -> List[DiagnosticIssue]:
    """Check Python runtime compatibility."""
    current = ".".join(str(part) for part in sys.version_info[:3])
    minimum = ".".join(str(part) for part in MIN_PYTHON_VERSION)
    is_supported = sys.version_info[:2] >= MIN_PYTHON_VERSION

    if is_supported:
        return [
            make_issue(
                name="Python Runtime",
                status=f"PASS - {current}",
                severity="info",
                check_id="python_runtime",
                category="runtime",
                details=f"Minimum supported version is {minimum}",
                confidence="high",
                evidence=[f"sys.version_info={sys.version_info[:3]}"],
                metadata={"current_version": current, "minimum_version": minimum},
            )
        ]

    return [
        make_issue(
            name="Python Runtime",
            status=f"FAIL - Unsupported version ({current})",
            severity="critical",
            fix=f"Upgrade Python to >={minimum}",
            check_id="python_runtime",
            category="runtime",
            details=f"Detected Python {current}; minimum supported version is {minimum}",
            confidence="high",
            evidence=[f"sys.version_info={sys.version_info[:3]}"],
            metadata={"current_version": current, "minimum_version": minimum},
        )
    ]


def check_cuda_driver() -> List[DiagnosticIssue]:
    """Check NVIDIA CUDA driver availability."""
    if not check_command_exists("nvidia-smi"):
        platform_hint = _platform_hint()
        if platform_hint == "macos":
            return [
                make_issue(
                    name="NVIDIA GPU Driver",
                    status="INFO - NVIDIA CUDA driver tooling not detected (expected on most macOS systems)",
                    severity="info",
                    fix=(
                        "python -c \"import torch; print(getattr(torch.backends, 'mps', None) and "
                        'torch.backends.mps.is_available())"'
                    ),
                    check_id="cuda_driver",
                    category="gpu",
                    details="macOS typically relies on Apple Silicon MPS or CPU workflows instead of CUDA.",
                    confidence="high",
                    evidence=["platform=macos", "nvidia-smi missing from PATH"],
                )
            ]
        if platform_hint == "wsl2":
            return [
                make_issue(
                    name="NVIDIA GPU Driver",
                    status="WARN - nvidia-smi not found in WSL",
                    severity="warning",
                    fix="Check host GPU drivers and WSL GPU passthrough, then run: nvidia-smi",
                    check_id="cuda_driver",
                    category="gpu",
                    details="WSL environment detected. GPU passthrough may not be active in this distro/session.",
                    confidence="high",
                    evidence=["platform=wsl2", "nvidia-smi missing from PATH"],
                )
            ]
        return [
            make_issue(
                name="NVIDIA GPU Driver",
                status="FAIL - nvidia-smi not found",
                severity="critical",
                fix="Install NVIDIA drivers, then verify with: nvidia-smi",
                check_id="cuda_driver",
                category="gpu",
                details="The NVIDIA CLI utility is missing from PATH",
                confidence="high",
                evidence=["nvidia-smi missing from PATH"],
            )
        ]

    try:
        result = run_command(["nvidia-smi"], timeout=10)
    except Exception as exc:
        return [
            make_issue(
                name="NVIDIA GPU Driver",
                status="FAIL - nvidia-smi execution failed",
                severity="critical",
                fix="Reinstall NVIDIA drivers, reboot, then run: nvidia-smi",
                check_id="cuda_driver",
                category="gpu",
                details=_summarize_exception(exc),
                confidence="medium",
                evidence=[_summarize_exception(exc)],
            )
        ]

    if result.returncode != 0:
        return [
            make_issue(
                name="NVIDIA GPU Driver",
                status="FAIL - nvidia-smi returned an error",
                severity="critical",
                fix="Reinstall NVIDIA drivers, reboot, then run: nvidia-smi",
                check_id="cuda_driver",
                category="gpu",
                details=(result.stderr or result.stdout or "").strip() or None,
                confidence="high",
                evidence=[f"returncode={result.returncode}"],
            )
        ]

    output = result.stdout
    cuda_match = re.search(r"CUDA Version: (\d+\.\d+)", output)
    driver_match = re.search(r"Driver Version: ([^\s]+)", output)
    gpu_names = re.findall(r"\|\s+\d+\s+([^|]+?)\s{2,}", output)
    cuda_version = cuda_match.group(1) if cuda_match else "unknown"
    driver_version = driver_match.group(1) if driver_match else "unknown"
    details = f"Driver {driver_version}, CUDA {cuda_version}"
    if gpu_names:
        details += f", GPUs: {', '.join(name.strip() for name in gpu_names[:2])}"

    return [
        make_issue(
            name="NVIDIA GPU Driver",
            status=f"PASS - CUDA {cuda_version}",
            severity="info",
            check_id="cuda_driver",
            category="gpu",
            details=details,
            confidence="high",
            evidence=[f"driver_version={driver_version}", f"cuda_version={cuda_version}"],
            metadata={"driver_version": driver_version, "cuda_version": cuda_version},
        )
    ]


def check_pytorch_cuda() -> List[DiagnosticIssue]:
    """Check PyTorch installation, CUDA visibility, and a small runtime execution path."""
    if torch is None:
        return [
            make_issue(
                name="PyTorch Installation",
                status="FAIL - Not installed",
                severity="critical",
                fix="pip install torch --index-url https://download.pytorch.org/whl/cu124",
                check_id="pytorch_installation",
                category="pytorch",
                details="PyTorch import failed",
                confidence="high",
                evidence=["import torch failed"],
            )
        ]

    issues: List[DiagnosticIssue] = []
    torch_version = getattr(torch, "__version__", "unknown")
    version_ok = _is_version_at_least(torch_version, MIN_PYTORCH_VERSION)

    if version_ok is False:
        issues.append(
            make_issue(
                name="PyTorch Version",
                status=f"WARN - Old version ({torch_version})",
                severity="warning",
                fix=(
                    "pip install --upgrade "
                    "torch --index-url https://download.pytorch.org/whl/cu124"
                ),
                check_id="pytorch_version",
                category="pytorch",
                details=f"Recommended version is >= {MIN_PYTORCH_VERSION}",
                confidence="high",
                evidence=[f"torch.__version__={torch_version}"],
                metadata={"current_version": torch_version, "minimum_version": MIN_PYTORCH_VERSION},
            )
        )
    else:
        issues.append(
            make_issue(
                name="PyTorch Version",
                status=f"PASS - {torch_version}",
                severity="info",
                check_id="pytorch_version",
                category="pytorch",
                details=f"Recommended baseline is >= {MIN_PYTORCH_VERSION}",
                confidence="high",
                evidence=[f"torch.__version__={torch_version}"],
            )
        )

    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:
        issues.append(
            make_issue(
                name="PyTorch CUDA",
                status="FAIL - CUDA availability check failed",
                severity="critical",
                fix="Reinstall PyTorch with the correct CUDA wheel for your system",
                check_id="pytorch_cuda",
                category="pytorch",
                details=_summarize_exception(exc),
                confidence="medium",
                evidence=[_summarize_exception(exc)],
            )
        )
        return issues

    if not cuda_available:
        torch_cuda_build = getattr(torch.version, "cuda", None)
        issues.append(
            make_issue(
                name="PyTorch CUDA",
                status="FAIL - CUDA not available",
                severity="critical",
                fix="pip install torch --index-url https://download.pytorch.org/whl/cu124",
                check_id="pytorch_cuda",
                category="pytorch",
                details=f"PyTorch {torch_version} is installed but torch.cuda.is_available() is false",
                likely_cause=(
                    "The installed PyTorch build does not include CUDA support."
                    if torch_cuda_build is None
                    else "The installed PyTorch build cannot use the currently available CUDA runtime."
                ),
                confidence="high",
                evidence=[
                    "torch.cuda.is_available() == False",
                    f"torch.version.cuda={torch_cuda_build or 'None'}",
                ],
                metadata={"torch_version": torch_version, "torch_cuda_build": torch_cuda_build},
            )
        )
        return issues

    cuda_version = getattr(torch.version, "cuda", None) or "unknown"
    device_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0) if device_count > 0 else "unknown"
    issues.append(
        make_issue(
            name="PyTorch CUDA",
            status=f"PASS - CUDA {cuda_version} ({device_count} GPU(s))",
            severity="info",
            check_id="pytorch_cuda",
            category="pytorch",
            details=f"Primary device: {device_name}",
            confidence="high",
            evidence=[
                f"torch.version.cuda={cuda_version}",
                f"torch.cuda.device_count()={device_count}",
            ],
            metadata={
                "torch_version": torch_version,
                "cuda_version": cuda_version,
                "device_count": device_count,
                "device_name": device_name,
            },
        )
    )

    try:
        device = torch.device("cuda:0")
        a = torch.tensor([1.0, 2.0], device=device)
        b = torch.tensor([3.0, 4.0], device=device)
        result = (a + b).tolist()
        issues.append(
            make_issue(
                name="PyTorch CUDA Execution",
                status="PASS - Tensor operation succeeded",
                severity="info",
                check_id="pytorch_cuda_execution",
                category="pytorch",
                details=f"Sample add result: {result}",
                confidence="high",
                evidence=[f"cuda_tensor_add={result}"],
            )
        )
    except Exception as exc:
        issues.append(
            make_issue(
                name="PyTorch CUDA Execution",
                status="FAIL - Tensor operation failed",
                severity="critical",
                fix="Reinstall the matching CUDA-enabled PyTorch build and verify the NVIDIA driver",
                check_id="pytorch_cuda_execution",
                category="pytorch",
                details=_summarize_exception(exc),
                confidence="high",
                evidence=[_summarize_exception(exc)],
            )
        )

    return issues


def _library_issue(lib_name: str, minimum: str) -> DiagnosticIssue:
    """Build a normalized library issue."""
    try:
        module = importlib.import_module(lib_name)
        version = getattr(module, "__version__", "unknown")
    except ImportError:
        return make_issue(
            name=lib_name,
            status="FAIL - Not installed",
            severity="critical",
            fix=f"pip install {lib_name}>={minimum}",
            check_id=f"library_{lib_name}",
            category="dependencies",
            details=f"{lib_name} is required for the recommended training stack",
            confidence="high",
            evidence=[f"import {lib_name} failed"],
            metadata={"minimum_version": minimum},
        )

    version_ok = _is_version_at_least(version, minimum)
    if version_ok is False:
        return make_issue(
            name=lib_name,
            status=f"WARN - Old version ({version})",
            severity="warning",
            fix=f"pip install --upgrade {lib_name}>={minimum}",
            check_id=f"library_{lib_name}",
            category="dependencies",
            details=f"Recommended version is >= {minimum}",
            confidence="high",
            evidence=[f"{lib_name}.__version__={version}"],
            metadata={"current_version": version, "minimum_version": minimum},
        )

    return make_issue(
        name=lib_name,
        status=f"PASS - {version}",
        severity="info",
        check_id=f"library_{lib_name}",
        category="dependencies",
        details=f"Recommended version is >= {minimum}",
        confidence="high" if version_ok is not None else "medium",
        evidence=[f"{lib_name}.__version__={version}"],
        metadata={"current_version": version, "minimum_version": minimum},
    )


def check_ml_libraries() -> List[DiagnosticIssue]:
    """Check common ML library installations."""
    required_libs = {
        "transformers": MIN_TRANSFORMERS_VERSION,
        "peft": MIN_PEFT_VERSION,
        "trl": MIN_TRL_VERSION,
        "datasets": MIN_DATASETS_VERSION,
        "accelerate": MIN_ACCELERATE_VERSION,
    }
    return [_library_issue(name, minimum) for name, minimum in required_libs.items()]


def check_tensorflow_keras() -> List[DiagnosticIssue]:
    """Check TensorFlow and Keras readiness."""
    issues: List[DiagnosticIssue] = []
    tensorflow = None
    tf_version = "unknown"

    try:
        tensorflow = importlib.import_module("tensorflow")
        tf_version = getattr(tensorflow, "__version__", "unknown")
        gpu_devices = []
        if hasattr(tensorflow, "config"):
            try:
                gpu_devices = list(tensorflow.config.list_physical_devices("GPU"))
            except Exception as exc:
                issues.append(
                    make_issue(
                        name="TensorFlow Device Discovery",
                        status="WARN - GPU enumeration failed",
                        severity="warning",
                        fix="Check your TensorFlow CUDA installation and driver/toolkit compatibility",
                        check_id="tensorflow_device_discovery",
                        category="tensorflow",
                        details=_summarize_exception(exc),
                        confidence="medium",
                        evidence=[_summarize_exception(exc)],
                    )
                )

        status = f"PASS - {tf_version}"
        severity = "info"
        fix = ""
        details = (
            f"Detected {len(gpu_devices)} GPU device(s)"
            if gpu_devices
            else "CPU-only TensorFlow runtime"
        )
        if check_command_exists("nvidia-smi") and not gpu_devices:
            status = f"WARN - {tf_version} (GPU not visible)"
            severity = "warning"
            fix = "pip install --upgrade tensorflow[and-cuda]"

        issues.append(
            make_issue(
                name="TensorFlow",
                status=status,
                severity=severity,
                fix=fix,
                check_id="tensorflow_runtime",
                category="tensorflow",
                details=details,
                confidence="high",
                evidence=[f"tensorflow.__version__={tf_version}", f"gpu_count={len(gpu_devices)}"],
                metadata={"version": tf_version, "gpu_count": len(gpu_devices)},
            )
        )

        try:
            a = tensorflow.constant([[1.0, 2.0]])
            b = tensorflow.constant([[3.0], [4.0]])
            result_value = tensorflow.matmul(a, b).numpy()
            result = result_value.tolist() if hasattr(result_value, "tolist") else result_value
            issues.append(
                make_issue(
                    name="TensorFlow Execution",
                    status="PASS - Basic tensor execution succeeded",
                    severity="info",
                    check_id="tensorflow_execution",
                    category="tensorflow",
                    details=f"Sample matmul result: {result}",
                    confidence="high",
                    evidence=[f"matmul_result={result}"],
                )
            )
        except Exception as exc:
            issues.append(
                make_issue(
                    name="TensorFlow Execution",
                    status="WARN - Basic tensor execution failed",
                    severity="warning",
                    fix="Reinstall TensorFlow and verify native dependencies",
                    check_id="tensorflow_execution",
                    category="tensorflow",
                    details=_summarize_exception(exc),
                    confidence="medium",
                    evidence=[_summarize_exception(exc)],
                )
            )

    except ImportError:
        issues.append(
            make_issue(
                name="TensorFlow",
                status="INFO - Not installed",
                severity="info",
                fix="pip install tensorflow[and-cuda]",
                check_id="tensorflow_runtime",
                category="tensorflow",
                details="Optional unless you use TensorFlow/Keras workloads",
                confidence="high",
                evidence=["import tensorflow failed"],
            )
        )
    except Exception as exc:
        issues.append(
            make_issue(
                name="TensorFlow",
                status="WARN - Import succeeded but runtime initialization failed",
                severity="warning",
                fix="Inspect TensorFlow runtime logs and verify CUDA/cuDNN compatibility",
                check_id="tensorflow_runtime",
                category="tensorflow",
                details=_summarize_exception(exc),
                confidence="medium",
                evidence=[_summarize_exception(exc)],
            )
        )

    try:
        keras = importlib.import_module("keras")
        keras_version = getattr(keras, "__version__", "unknown")
        keras_ok = _is_version_at_least(keras_version, "3.0.0")
        if keras_ok is False:
            issues.append(
                make_issue(
                    name="Keras",
                    status=f"WARN - Old version ({keras_version})",
                    severity="warning",
                    fix="pip install --upgrade keras>=3.0.0",
                    check_id="keras_version",
                    category="tensorflow",
                    details="Keras 3 is recommended for modern TensorFlow workflows",
                    confidence="high",
                    evidence=[f"keras.__version__={keras_version}"],
                )
            )
        else:
            issues.append(
                make_issue(
                    name="Keras",
                    status=f"PASS - {keras_version}",
                    severity="info",
                    check_id="keras_version",
                    category="tensorflow",
                    details="Keras 3 ready" if keras_version != "unknown" else "Installed",
                    confidence="high" if keras_ok is not None else "medium",
                    evidence=[f"keras.__version__={keras_version}"],
                )
            )
    except ImportError:
        severity = "warning" if tensorflow is not None else "info"
        status = "WARN - Not installed" if tensorflow is not None else "INFO - Not installed"
        details = (
            f"TensorFlow {tf_version} is installed without standalone Keras 3"
            if tensorflow is not None
            else "Optional unless you use Keras directly"
        )
        issues.append(
            make_issue(
                name="Keras",
                status=status,
                severity=severity,
                fix="pip install keras>=3.0.0",
                check_id="keras_version",
                category="tensorflow",
                details=details,
                confidence="high",
                evidence=["import keras failed"],
            )
        )

    return issues


def check_jax_flax() -> List[DiagnosticIssue]:
    """Check JAX/Flax installation and backend visibility."""
    issues: List[DiagnosticIssue] = []

    try:
        jax = importlib.import_module("jax")
        jax_version = getattr(jax, "__version__", "unknown")
        devices = list(jax.devices())
        backend = jax.default_backend()
        device_kinds = sorted({getattr(device, "platform", "unknown") for device in devices})
        details = (
            f"Backend: {backend}, devices: {len(devices)} "
            f"({', '.join(device_kinds) if device_kinds else 'none'})"
        )
        severity = "info"
        status = f"PASS - {jax_version}"
        fix = ""
        if check_command_exists("nvidia-smi") and backend == "cpu":
            severity = "warning"
            status = f"WARN - {jax_version} (CPU backend)"
            fix = 'pip install --upgrade "jax[cuda]"'

        issues.append(
            make_issue(
                name="JAX",
                status=status,
                severity=severity,
                fix=fix,
                check_id="jax_runtime",
                category="jax",
                details=details,
                confidence="high",
                evidence=[f"jax.__version__={jax_version}", f"backend={backend}"],
                metadata={"version": jax_version, "backend": backend, "device_count": len(devices)},
            )
        )

        try:
            array = jax.numpy.array([1.0, 2.0, 3.0])
            issues.append(
                make_issue(
                    name="JAX Execution",
                    status="PASS - Basic array execution succeeded",
                    severity="info",
                    check_id="jax_execution",
                    category="jax",
                    details=f"Array sum: {float(array.sum())}",
                    confidence="high",
                    evidence=[f"array_sum={float(array.sum())}"],
                )
            )
        except Exception as exc:
            issues.append(
                make_issue(
                    name="JAX Execution",
                    status="WARN - Basic array execution failed",
                    severity="warning",
                    fix="Reinstall JAX and verify the selected accelerator backend",
                    check_id="jax_execution",
                    category="jax",
                    details=_summarize_exception(exc),
                    confidence="medium",
                    evidence=[_summarize_exception(exc)],
                )
            )
    except ImportError:
        issues.append(
            make_issue(
                name="JAX",
                status="INFO - Not installed",
                severity="info",
                fix="pip install jax flax",
                check_id="jax_runtime",
                category="jax",
                details="Optional unless you use JAX/Flax workloads",
                confidence="high",
                evidence=["import jax failed"],
            )
        )

    try:
        flax = importlib.import_module("flax")
        flax_version = getattr(flax, "__version__", "unknown")
        issues.append(
            make_issue(
                name="Flax",
                status=f"PASS - {flax_version}",
                severity="info",
                check_id="flax_runtime",
                category="jax",
                confidence="high",
                evidence=[f"flax.__version__={flax_version}"],
            )
        )
    except ImportError:
        issues.append(
            make_issue(
                name="Flax",
                status="INFO - Not installed",
                severity="info",
                fix="pip install flax",
                check_id="flax_runtime",
                category="jax",
                details="Install if you train JAX/Flax models",
                confidence="high",
                evidence=["import flax failed"],
            )
        )

    return issues


def check_gpu_memory() -> List[DiagnosticIssue]:
    """Check GPU memory availability."""
    if torch is None or not torch.cuda.is_available():
        return []

    try:
        free_mem, total_mem = torch.cuda.mem_get_info(0)
        free_gb = free_mem / (1024**3)
        total_gb = total_mem / (1024**3)
    except Exception as exc:
        return [
            make_issue(
                name="GPU Memory",
                status="WARN - Memory query failed",
                severity="warning",
                fix="Check GPU usage with: nvidia-smi",
                check_id="gpu_memory",
                category="gpu",
                details=_summarize_exception(exc),
                confidence="medium",
                evidence=[_summarize_exception(exc)],
            )
        ]

    if free_gb < MIN_GPU_MEMORY_GB:
        return [
            make_issue(
                name="GPU Memory",
                status=f"WARN - Low memory ({free_gb:.1f}GB free)",
                severity="warning",
                fix="Close other GPU processes or use a smaller model/batch size",
                check_id="gpu_memory",
                category="gpu",
                details=f"Free: {free_gb:.1f}GB / Total: {total_gb:.1f}GB",
                confidence="high",
                evidence=[f"free_gb={free_gb:.2f}", f"total_gb={total_gb:.2f}"],
                metadata={"free_gb": round(free_gb, 2), "total_gb": round(total_gb, 2)},
            )
        ]

    return [
        make_issue(
            name="GPU Memory",
            status=f"PASS - {free_gb:.1f}GB free",
            severity="info",
            check_id="gpu_memory",
            category="gpu",
            details=f"Free: {free_gb:.1f}GB / Total: {total_gb:.1f}GB",
            confidence="high",
            evidence=[f"free_gb={free_gb:.2f}", f"total_gb={total_gb:.2f}"],
            metadata={"free_gb": round(free_gb, 2), "total_gb": round(total_gb, 2)},
        )
    ]


def check_disk_space() -> List[DiagnosticIssue]:
    """Check disk space for model caches."""
    try:
        import shutil

        cache_root = get_home_config_dir().parent / ".cache"
        stat = shutil.disk_usage(cache_root)
        free_gb = stat.free / (1024**3)
    except Exception as exc:
        return [
            make_issue(
                name="Disk Space",
                status="WARN - Disk space check failed",
                severity="warning",
                fix="Check disk space manually for your cache drive",
                check_id="disk_space",
                category="system",
                details=_summarize_exception(exc),
                confidence="medium",
                evidence=[_summarize_exception(exc)],
            )
        ]

    if free_gb < MIN_DISK_SPACE_GB:
        return [
            make_issue(
                name="Disk Space",
                status=f"WARN - Low space ({free_gb:.1f}GB free)",
                severity="warning",
                fix="Free up disk space or move large caches to a different drive",
                check_id="disk_space",
                category="system",
                details=f"Checked: {cache_root} | Free: {format_size(stat.free)}",
                confidence="high",
                evidence=[f"cache_root={cache_root}", f"free_gb={free_gb:.2f}"],
                metadata={"free_gb": round(free_gb, 2), "cache_root": str(cache_root)},
            )
        ]

    return [
        make_issue(
            name="Disk Space",
            status=f"PASS - {free_gb:.1f}GB free",
            severity="info",
            check_id="disk_space",
            category="system",
            details=f"Checked: {cache_root} | Free: {format_size(stat.free)}",
            confidence="high",
            evidence=[f"cache_root={cache_root}", f"free_gb={free_gb:.2f}"],
            metadata={"free_gb": round(free_gb, 2), "cache_root": str(cache_root)},
        )
    ]


def check_docker_gpu() -> List[DiagnosticIssue]:
    """Check Docker GPU support."""
    if not check_command_exists("docker"):
        return [
            make_issue(
                name="Docker GPU Support",
                status="INFO - Docker not installed",
                severity="info",
                fix="Install Docker, then test with: docker run --rm hello-world",
                check_id="docker_gpu",
                category="docker",
                details="Docker CLI was not found in PATH",
                confidence="high",
                evidence=["docker missing from PATH"],
            )
        ]

    try:
        info_result = run_command(["docker", "info"], timeout=15)
        if info_result.returncode != 0:
            return [
                make_issue(
                    name="Docker GPU Support",
                    status="WARN - Docker daemon unavailable",
                    severity="warning",
                    fix="Start Docker and rerun diagnostics",
                    check_id="docker_gpu",
                    category="docker",
                    details=(info_result.stderr or info_result.stdout or "").strip() or None,
                    confidence="high",
                    evidence=[f"docker info returncode={info_result.returncode}"],
                )
            ]
    except Exception as exc:
        return [
            make_issue(
                name="Docker GPU Support",
                status="INFO - Docker GPU test skipped",
                severity="info",
                fix=(
                    "Test manually: docker run --rm --gpus all "
                    "nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi"
                ),
                check_id="docker_gpu",
                category="docker",
                details=_summarize_exception(exc),
                confidence="low",
                evidence=[_summarize_exception(exc)],
            )
        ]

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
    except Exception as exc:
        return [
            make_issue(
                name="Docker GPU Support",
                status="INFO - Docker GPU test skipped",
                severity="info",
                fix=(
                    "Test manually: docker run --rm --gpus all "
                    "nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi"
                ),
                check_id="docker_gpu",
                category="docker",
                details=_summarize_exception(exc),
                confidence="low",
                evidence=[_summarize_exception(exc)],
            )
        ]

    if result.returncode == 0:
        return [
            make_issue(
                name="Docker GPU Support",
                status="PASS - GPU available in Docker",
                severity="info",
                check_id="docker_gpu",
                category="docker",
                details="`docker run --gpus all ... nvidia-smi` completed successfully",
                confidence="high",
                evidence=["docker run test succeeded"],
            )
        ]

    return [
        make_issue(
            name="Docker GPU Support",
            status="FAIL - GPU not accessible in Docker",
            severity="warning",
            fix=(
                "Install nvidia-container-toolkit, then test: docker run --rm --gpus all "
                "nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi"
            ),
            check_id="docker_gpu",
            category="docker",
            details=(result.stderr or result.stdout or "").strip() or None,
            confidence="high",
            evidence=[f"docker run returncode={result.returncode}"],
        )
    ]


def check_internet_connectivity() -> List[DiagnosticIssue]:
    """Check internet connectivity for HF Hub."""
    from .retry import retry_network

    @retry_network
    def _check_connectivity() -> None:
        import urllib.request

        with urllib.request.urlopen("https://huggingface.co", timeout=5):
            return None

    try:
        _check_connectivity()
        return [
            make_issue(
                name="Internet Connectivity",
                status="PASS - HF Hub accessible",
                severity="info",
                check_id="internet_connectivity",
                category="network",
                confidence="high",
                evidence=["https://huggingface.co reachable"],
            )
        ]
    except Exception as exc:
        return [
            make_issue(
                name="Internet Connectivity",
                status="WARN - Cannot reach HF Hub",
                severity="warning",
                fix=(
                    "Check connectivity with: python -c "
                    "\"import urllib.request; urllib.request.urlopen('https://huggingface.co', timeout=5)\""
                ),
                check_id="internet_connectivity",
                category="network",
                details=_summarize_exception(exc),
                confidence="medium",
                evidence=[_summarize_exception(exc)],
            )
        ]


def _run_check_group(
    checks: List[Callable[[], List[DiagnosticIssue]]],
    *,
    parallel: bool,
    timeout: float,
    failure_severity: str,
) -> List[DiagnosticIssue]:
    """Run a check group either sequentially or in parallel."""
    from .parallel import run_parallel_with_results

    issues: List[DiagnosticIssue] = []

    if parallel:
        results = run_parallel_with_results(
            lambda check_func: check_func(),
            checks,
            max_workers=len(checks),
            timeout=timeout,
        )
        for check_func, result in results:
            if isinstance(result, Exception):
                issues.append(
                    make_issue(
                        name=check_func.__name__.replace("check_", "").replace("_", " ").title(),
                        status="FAIL - Check error",
                        severity=failure_severity,
                        fix="Run diagnostics again with: mlenvdoctor diagnose --log-level DEBUG",
                        check_id=f"internal_{check_func.__name__}",
                        category="internal",
                        details=_summarize_exception(result),
                        confidence="medium",
                        evidence=[_summarize_exception(result)],
                    )
                )
            else:
                issues.extend(result)
        return issues

    for check_func in checks:
        try:
            issues.extend(check_func())
        except Exception as exc:
            issues.append(
                make_issue(
                    name=check_func.__name__.replace("check_", "").replace("_", " ").title(),
                    status="FAIL - Check error",
                    severity=failure_severity,
                    fix="Run diagnostics again with: mlenvdoctor diagnose --log-level DEBUG",
                    check_id=f"internal_{check_func.__name__}",
                    category="internal",
                    details=_summarize_exception(exc),
                    confidence="medium",
                    evidence=[_summarize_exception(exc)],
                )
            )
    return issues


def diagnose_env(
    full: bool = False,
    parallel: bool = True,
    show_header: bool = True,
) -> List[DiagnosticIssue]:
    """Run diagnostic checks and return normalized issues."""
    if show_header:
        console.print(
            f"[bold blue]{icon_search()} Running ML Environment Diagnostics...[/bold blue]\n"
        )

    core_checks = [
        check_python_runtime,
        check_accelerator_backend,
        check_cuda_driver,
        check_pytorch_cuda,
        check_ml_libraries,
        check_tensorflow_keras,
        check_jax_flax,
    ]
    issues = _run_check_group(
        core_checks, parallel=parallel, timeout=60.0, failure_severity="critical"
    )

    if full:
        extended_checks = [
            check_gpu_memory,
            check_disk_space,
            check_docker_gpu,
            check_internet_connectivity,
        ]
        issues.extend(
            _run_check_group(
                extended_checks, parallel=parallel, timeout=120.0, failure_severity="warning"
            )
        )

    issues.extend(_compatibility_matrix_issues(issues))

    return issues


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

    critical_count = sum(1 for i in issues if i.severity == "critical" and "FAIL" in i.status)
    warning_count = sum(
        1 for i in issues if i.severity == "warning" and ("WARN" in i.status or "FAIL" in i.status)
    )
    pass_count = sum(1 for i in issues if "PASS" in i.status)

    console.print()
    console.print(f"[green]{icon_check()} Passed: {pass_count}[/green]")
    if warning_count > 0:
        console.print(f"[yellow]{icon_warning()}  Warnings: {warning_count}[/yellow]")
    if critical_count > 0:
        console.print(f"[red]{icon_cross()} Critical Issues: {critical_count}[/red]")

    if critical_count == 0 and warning_count == 0:
        console.print("\n[bold green]Your ML environment looks ready for fine-tuning![/bold green]")
    elif critical_count > 0:
        console.print(
            f"\n[bold red]{icon_warning()}  Please fix critical issues before proceeding.[/bold red]"
        )
    else:
        console.print(
            "\n[bold yellow]Consider addressing warnings for optimal performance.[/bold yellow]"
        )
