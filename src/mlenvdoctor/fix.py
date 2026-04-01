"""Auto-fix and requirements generation for ML Environment Doctor."""

import sys
from pathlib import Path
from typing import Optional

from .diagnose import diagnose_env
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


def get_stack_requirements(stack: str) -> list[str]:
    """Return the requirement list for a named stack."""
    if stack not in ML_STACKS:
        print_error(f"Unknown stack: {stack}. Available: {list(ML_STACKS.keys())}")
        sys.exit(1)
    return ML_STACKS[stack]


def generate_requirements_txt(
    stack: str = "trl-peft", output_file: str = "requirements-mlenvdoctor.txt"
) -> Path:
    """Generate requirements.txt file."""
    requirements = get_stack_requirements(stack)
    output_path = Path(output_file)

    try:
        import torch

        if torch.cuda.is_available():
            content = "# PyTorch with CUDA 12.4\n"
            content += (
                "# Install with: pip install torch --index-url "
                "https://download.pytorch.org/whl/cu124\n"
            )
            content += f"# Then: pip install -r {output_path.name}\n\n"
        else:
            content = "# Standard PyTorch (CPU or CUDA)\n\n"
    except ImportError:
        content = "# PyTorch installation\n"
        content += "# For CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu124\n"
        content += "# For CPU: pip install torch\n\n"

    content += "\n".join(requirements)
    content += "\n"

    output_path.write_text(content, encoding="utf-8")
    print_success(f"Generated {output_path}")
    return output_path


def generate_conda_env(
    stack: str = "trl-peft", output_file: str = "environment-mlenvdoctor.yml"
) -> Path:
    """Generate conda environment file."""
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
        import torch

        if not torch.cuda.is_available():
            print_info("Installing PyTorch with CUDA support...")
            try:
                result = run_command(
                    [
                        python_cmd,
                        "-m",
                        "pip",
                        "install",
                        "torch",
                        "--index-url",
                        "https://download.pytorch.org/whl/cu124",
                    ],
                    timeout=600,
                )
                if result.returncode == 0:
                    print_success("PyTorch with CUDA installed")
            except Exception as e:
                print_warning(f"PyTorch CUDA installation skipped: {e}")
    except ImportError:
        print_info("Installing PyTorch with CUDA support...")
        try:
            result = run_command(
                [
                    python_cmd,
                    "-m",
                    "pip",
                    "install",
                    "torch",
                    "--index-url",
                    "https://download.pytorch.org/whl/cu124",
                ],
                timeout=600,
            )
            if result.returncode == 0:
                print_success("PyTorch with CUDA installed")
        except Exception as e:
            print_warning(f"PyTorch CUDA installation failed: {e}")

    try:
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
    except Exception as e:
        print_error(f"Installation error: {e}")
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
            activate_cmd = r".venv\Scripts\activate"
        else:
            activate_cmd = "source .venv/bin/activate"
        print_info(f"Activate with: {activate_cmd}")
        return env_path
    except Exception as e:
        print_error(f"Failed to create virtual environment: {e}")
        return None


def auto_fix(use_conda: bool = False, create_venv: bool = False, stack: str = "trl-peft") -> bool:
    """Auto-fix environment issues based on diagnostics."""
    console.print(f"[bold blue]{icon_wrench()} Running Auto-Fix...[/bold blue]\n")

    issues = diagnose_env(full=False)
    critical_issues = [issue for issue in issues if issue.severity == "critical" and "FAIL" in issue.status]

    if not critical_issues:
        print_success("No critical issues found! Environment looks good.")
        return True

    console.print(f"[yellow]Found {len(critical_issues)} critical issue(s) to fix[/yellow]\n")

    if use_conda:
        env_file = generate_conda_env(stack=stack)
        print_info("Conda environment file generated. Create environment manually:")
        console.print(f"[cyan]  conda env create -f {env_file}[/cyan]")
        return True

    req_file = generate_requirements_txt(stack=stack)
    venv_python: Optional[str] = None

    if create_venv:
        venv_path = create_virtualenv()
        if venv_path:
            venv_python = str(get_virtualenv_python(venv_path))
            print_info(f"Using virtual environment Python: {venv_python}")

    console.print()
    install = console.input("[bold yellow]Install requirements now? (y/n): [/bold yellow]")
    if install.lower() in ["y", "yes"]:
        return install_requirements(
            str(req_file),
            use_conda=use_conda,
            python_executable=venv_python,
        )

    print_info("Requirements file generated. Install manually with:")
    if venv_python:
        console.print(f"[cyan]  {venv_python} -m pip install -r {req_file}[/cyan]")
    else:
        console.print(f"[cyan]  pip install -r {req_file}[/cyan]")
    return True
