"""Auto-fix and requirements generation for ML Environment Doctor."""

import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .diagnose import DiagnosticIssue, diagnose_env
from .utils import (
    check_command_exists,
    console,
    print_error,
    print_info,
    print_success,
    print_warning,
    run_command,
)

# ML Stack definitions
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
}


def generate_requirements_txt(
    stack: str = "trl-peft", output_file: str = "requirements-mlenvdoctor.txt"
) -> Path:
    """Generate requirements.txt file."""
    if stack not in ML_STACKS:
        print_error(f"Unknown stack: {stack}. Available: {list(ML_STACKS.keys())}")
        sys.exit(1)

    requirements = ML_STACKS[stack]
    output_path = Path(output_file)

    # Check if CUDA is available to add PyTorch index URL
    try:
        import torch

        if torch.cuda.is_available():
            # Add comment about CUDA index
            content = "# PyTorch with CUDA 12.4\n"
            content += "# Install with: pip install torch --index-url https://download.pytorch.org/whl/cu124\n"
            content += "# Then: pip install -r requirements-mlenvdoctor.txt\n\n"
        else:
            content = "# Standard PyTorch (CPU or CUDA)\n\n"
    except ImportError:
        content = "# PyTorch installation\n"
        content += (
            "# For CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu124\n"
        )
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
    if stack not in ML_STACKS:
        print_error(f"Unknown stack: {stack}. Available: {list(ML_STACKS.keys())}")
        sys.exit(1)

    requirements = ML_STACKS[stack]
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

    # Filter out torch from pip requirements (it's in conda)
    pip_requirements = [r for r in requirements if not r.startswith("torch")]
    for req in pip_requirements:
        content += f"    - {req}\n"

    output_path.write_text(content, encoding="utf-8")
    print_success(f"Generated {output_path}")
    print_info(f"Create environment with: conda env create -f {output_file}")
    return output_path


def install_requirements(requirements_file: str, use_conda: bool = False) -> bool:
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
        # For conda, user should create env manually
        print_warning("Conda environment file generated. Please create environment manually.")
        return True

    print_info(f"Installing requirements from {requirements_file}...")

    # First install PyTorch with CUDA if needed
    try:
        import torch

        if not torch.cuda.is_available():
            print_info("Installing PyTorch with CUDA support...")
            try:
                result = run_command(
                    [
                        sys.executable,
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
                    sys.executable,
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

    # Install other requirements
    try:
        with console.status("[bold green]Installing requirements..."):
            result = run_command(
                [sys.executable, "-m", "pip", "install", "-r", requirements_file],
                timeout=600,
            )
            if result.returncode == 0:
                print_success("Requirements installed successfully!")
                return True
            else:
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
    console.print("[bold blue]🔧 Running Auto-Fix...[/bold blue]\n")

    # Run diagnostics
    issues = diagnose_env(full=False)
    critical_issues = [i for i in issues if i.severity == "critical" and "FAIL" in i.status]

    if not critical_issues:
        print_success("No critical issues found! Environment looks good.")
        return True

    console.print(f"[yellow]Found {len(critical_issues)} critical issue(s) to fix[/yellow]\n")

    # Generate requirements
    if use_conda:
        env_file = generate_conda_env(stack=stack)
        print_info("Conda environment file generated. Create environment manually:")
        console.print(f"[cyan]  conda env create -f {env_file}[/cyan]")
        return True
    else:
        req_file = generate_requirements_txt(stack=stack)

        if create_venv:
            venv_path = create_virtualenv()
            if venv_path:
                # Update pip command to use venv
                if sys.platform == "win32":
                    pip_cmd = str(venv_path / "Scripts" / "python.exe")
                else:
                    pip_cmd = str(venv_path / "bin" / "python")
                print_info(f"Using virtual environment Python: {pip_cmd}")

        # Offer to install
        console.print()
        install = console.input("[bold yellow]Install requirements now? (y/n): [/bold yellow]")
        if install.lower() in ["y", "yes"]:
            return install_requirements(str(req_file), use_conda=use_conda)
        else:
            print_info(f"Requirements file generated. Install manually with:")
            console.print(f"[cyan]  pip install -r {req_file}[/cyan]")
            return True
