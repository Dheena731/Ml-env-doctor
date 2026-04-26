"""Tests for diagnose module."""

from mlenvdoctor import diagnose as diagnose_module
from mlenvdoctor.diagnose import (
    DiagnosticIssue,
    check_accelerator_backend,
    check_cuda_driver,
    check_jax_flax,
    check_tensorflow_keras,
    diagnose_env,
    get_fix_commands,
    summarize_for_doctor,
)
from mlenvdoctor.parallel import run_parallel_with_results


def test_diagnostic_issue():
    """Test DiagnosticIssue class."""
    issue = DiagnosticIssue(
        name="Test Issue",
        status="FAIL",
        severity="critical",
        fix="Fix it",
        details="Details",
    )
    assert issue.name == "Test Issue"
    assert issue.status == "FAIL"
    assert issue.severity == "critical"
    assert issue.fix == "Fix it"
    assert issue.details == "Details"

    row = issue.to_row()
    assert len(row) == 4
    assert row[0] == "Test Issue"


def test_diagnose_env():
    """Test diagnose_env function."""
    issues = diagnose_env(full=False)
    assert isinstance(issues, list)
    assert all(isinstance(issue, DiagnosticIssue) for issue in issues)

    # Should have at least some issues
    assert len(issues) > 0


def test_diagnose_env_full():
    """Test full diagnose_env."""
    issues = diagnose_env(full=True)
    assert isinstance(issues, list)
    assert all(isinstance(issue, DiagnosticIssue) for issue in issues)
    # Full scan should return more issues
    assert len(issues) >= 3  # At least basic checks


def test_parallel_results_preserve_input_order():
    """Parallel helper should keep results aligned with the input order."""

    def work(item: int) -> int:
        if item == 1:
            import time

            time.sleep(0.05)
        return item * 10

    results = run_parallel_with_results(work, [1, 2, 3], max_workers=3, timeout=5.0)

    assert [item for item, _ in results] == [1, 2, 3]


def test_get_fix_commands_filters_actionable_issues():
    """Only issues with fix commands should be returned for machine-facing consumers."""
    issues = [
        DiagnosticIssue("A", "FAIL - bad", "critical", "pip install a"),
        DiagnosticIssue("B", "PASS - ok", "info", ""),
    ]

    fixes = get_fix_commands(issues)

    assert fixes == [
        {
            "issue": "A",
            "severity": "critical",
            "status": "FAIL - bad",
            "command": "pip install a",
            "check_id": "",
            "category": "general",
        }
    ]


def test_check_tensorflow_keras_installed(monkeypatch):
    """TensorFlow/Keras diagnostics should surface installed runtimes."""

    class FakeTensor:
        def __init__(self, value):
            self._value = value

        def numpy(self):
            return self._value

    class FakeTF:
        __version__ = "2.16.1"

        class config:
            @staticmethod
            def list_physical_devices(kind: str):
                return [object()] if kind == "GPU" else []

        @staticmethod
        def constant(value):
            return value

        @staticmethod
        def matmul(a, b):
            result = [[sum(a[0][i] * b[i][0] for i in range(len(b)))]]
            return FakeTensor(result)

    class FakeKeras:
        __version__ = "3.2.0"

    def fake_import_module(name: str):
        if name == "tensorflow":
            return FakeTF
        if name == "keras":
            return FakeKeras
        raise ImportError(name)

    monkeypatch.setattr(diagnose_module.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(diagnose_module, "check_command_exists", lambda cmd: cmd == "nvidia-smi")

    issues = check_tensorflow_keras()

    assert any(issue.name == "TensorFlow" and issue.status.startswith("PASS") for issue in issues)
    assert any(issue.name == "Keras" and issue.status.startswith("PASS") for issue in issues)
    assert any(
        issue.name == "TensorFlow Execution" and issue.status.startswith("PASS") for issue in issues
    )


def test_check_jax_flax_cpu_warning(monkeypatch):
    """JAX should warn when only CPU is visible while NVIDIA tooling exists."""

    class FakeDevice:
        platform = "cpu"

    class FakeJax:
        __version__ = "0.4.30"

        class numpy:
            @staticmethod
            def array(values):
                class FakeArray:
                    def __init__(self, items):
                        self.items = items

                    def sum(self):
                        return sum(self.items)

                return FakeArray(values)

        @staticmethod
        def devices():
            return [FakeDevice()]

        @staticmethod
        def default_backend():
            return "cpu"

    class FakeFlax:
        __version__ = "0.8.5"

    def fake_import_module(name: str):
        if name == "jax":
            return FakeJax
        if name == "flax":
            return FakeFlax
        raise ImportError(name)

    monkeypatch.setattr(diagnose_module.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(diagnose_module, "check_command_exists", lambda cmd: cmd == "nvidia-smi")

    issues = check_jax_flax()

    assert any(issue.name == "JAX" and issue.status.startswith("WARN") for issue in issues)
    assert any(issue.name == "Flax" and issue.status.startswith("PASS") for issue in issues)
    assert any(
        issue.name == "JAX Execution" and issue.status.startswith("PASS") for issue in issues
    )


def test_check_cuda_driver_is_platform_aware_on_macos(monkeypatch):
    """Missing nvidia-smi should be informational on macOS paths."""
    monkeypatch.setattr(diagnose_module, "check_command_exists", lambda cmd: False)
    monkeypatch.setattr(diagnose_module, "_platform_hint", lambda: "macos")

    issues = check_cuda_driver()

    assert len(issues) == 1
    assert issues[0].severity == "info"
    assert issues[0].status.startswith("INFO")


def test_check_cuda_driver_warns_for_wsl_passthrough(monkeypatch):
    """Missing nvidia-smi in WSL should point to passthrough guidance."""
    monkeypatch.setattr(diagnose_module, "check_command_exists", lambda cmd: False)
    monkeypatch.setattr(diagnose_module, "_platform_hint", lambda: "wsl2")

    issues = check_cuda_driver()

    assert len(issues) == 1
    assert issues[0].severity == "warning"
    assert "WSL" in issues[0].status


def test_check_accelerator_backend_warns_when_wsl_reports_cpu_with_nvidia(monkeypatch):
    """Backend detection should flag WSL CPU fallback when NVIDIA tooling is visible."""

    class FakeCuda:
        @staticmethod
        def is_available():
            return False

    class FakeBackends:
        mps = None

    class FakeTorch:
        cuda = FakeCuda()
        backends = FakeBackends()

    monkeypatch.setattr(diagnose_module, "torch", FakeTorch)
    monkeypatch.setattr(diagnose_module, "_platform_hint", lambda: "wsl2")
    monkeypatch.setattr(diagnose_module, "check_command_exists", lambda cmd: cmd == "nvidia-smi")

    issues = check_accelerator_backend()

    assert len(issues) == 1
    assert issues[0].severity == "warning"
    assert issues[0].status.startswith("WARN")
    assert issues[0].check_id == "accelerator_backend"


def test_summarize_for_doctor_includes_accelerator_backend_finding():
    """Backend warnings should surface as doctor-level root-cause findings."""
    issues = [
        DiagnosticIssue(
            name="Accelerator Backend",
            status="WARN - backend=cpu on Windows with NVIDIA tooling",
            severity="warning",
            fix="Use CUDA-enabled PyTorch",
            check_id="accelerator_backend",
            category="platform",
            likely_cause="Runtime backend mismatch",
            recommendation="Install CUDA build",
            verify_steps=['python -c "import torch; print(torch.cuda.is_available())"'],
            confidence="medium",
        )
    ]

    findings = summarize_for_doctor(issues)

    assert findings
    assert findings[0].check_id == "root_accelerator_backend"
    assert findings[0].category == "platform"


def test_summarize_for_doctor_prioritizes_actionable_issues():
    """Doctor summaries should include grouped root-cause style findings."""
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
        DiagnosticIssue(
            name="transformers",
            status="PASS - ok",
            severity="info",
            fix="",
            check_id="library_transformers",
            category="dependencies",
        ),
    ]

    findings = summarize_for_doctor(issues)

    assert len(findings) == 1
    assert findings[0].problem == "PyTorch cannot use the available GPU"
    assert findings[0].confidence == "high"
    assert findings[0].verify_steps
    assert "pytorch_cuda" in findings[0].linked_checks


def test_summarize_for_doctor_distinguishes_cpu_only_pytorch_build():
    """Doctor should distinguish a CPU-only torch build from a generic mismatch."""
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
            metadata={"torch_cuda_build": None},
        ),
    ]

    findings = summarize_for_doctor(issues)

    assert (
        findings[0].problem == "PyTorch is installed as a CPU-only build on a GPU-capable machine"
    )
    assert findings[0].check_id == "root_pytorch_cpu_only_build"


def test_summarize_for_doctor_groups_dependency_stack_issues():
    """Multiple dependency issues should collapse into one stack-level finding."""
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

    findings = summarize_for_doctor(issues)

    assert findings[0].problem == "The core ML training dependency stack is incomplete or outdated"
    assert findings[0].category == "dependencies"
    assert "library_transformers" in findings[0].linked_checks


def test_compatibility_matrix_flags_cpu_only_torch_on_gpu_machine():
    """Compatibility inference should emit machine-readable mismatch codes."""
    issues = [
        DiagnosticIssue(
            name="Python Runtime",
            status="PASS - 3.11.8",
            severity="info",
            fix="",
            check_id="python_runtime",
            category="runtime",
            metadata={"current_version": "3.11.8"},
        ),
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
            metadata={"torch_cuda_build": None},
        ),
    ]

    inferred = diagnose_module._compatibility_matrix_issues(issues)

    assert inferred
    assert inferred[0].check_id == "compatibility_matrix"
    assert inferred[0].mismatch_code == "PT_CPU_WHEEL_ON_GPU"
