"""Shared utilities for ML Environment Doctor."""

import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


def run_command(
    cmd: List[str],
    capture_output: bool = True,
    check: bool = False,
    timeout: Optional[int] = 30,
) -> subprocess.CompletedProcess:
    """Run a shell command with error handling."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=capture_output,
            text=True,
            check=check,
            timeout=timeout,
        )
        return result
    except subprocess.TimeoutExpired:
        console.print(f"[red]Command timed out: {' '.join(cmd)}[/red]")
        raise
    except FileNotFoundError:
        console.print(f"[red]Command not found: {cmd[0]}[/red]")
        raise
    except subprocess.CalledProcessError as e:
        if not check:
            return e  # type: ignore
        raise


def check_command_exists(cmd: str) -> bool:
    """Check if a command exists in PATH."""
    try:
        subprocess.run(
            [cmd, "--version"] if cmd != "nvidia-smi" else [cmd],
            capture_output=True,
            timeout=5,
            check=False,
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def get_home_config_dir() -> Path:
    """Get the configuration directory for mlenvdoctor."""
    home = Path.home()
    config_dir = home / ".mlenvdoctor"
    config_dir.mkdir(exist_ok=True)
    return config_dir


def print_success(message: str) -> None:
    """Print a success message."""
    console.print(f"[green]✅ {message}[/green]")


def print_error(message: str) -> None:
    """Print an error message."""
    console.print(f"[red]❌ {message}[/red]")


def print_warning(message: str) -> None:
    """Print a warning message."""
    console.print(f"[yellow]⚠️  {message}[/yellow]")


def print_info(message: str) -> None:
    """Print an info message."""
    console.print(f"[blue]ℹ️  {message}[/blue]")


def with_spinner(message: str):
    """Context manager for spinner during long operations."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    )


def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable size."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def get_python_version() -> Tuple[int, int, int]:
    """Get Python version as tuple."""
    return sys.version_info[:3]
