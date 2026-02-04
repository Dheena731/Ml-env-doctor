"""Shared utilities for ML Environment Doctor."""

import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import sys

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .exceptions import DiagnosticError
from .icons import icon_check, icon_cross, icon_info, icon_warning

# Configure console for Windows compatibility
if sys.platform == "win32":
    # Use legacy Windows renderer if needed
    console = Console(legacy_windows=True, force_terminal=True)
else:
    console = Console()


def run_command(
    cmd: List[str],
    capture_output: bool = True,
    check: bool = False,
    timeout: Optional[int] = 30,
) -> subprocess.CompletedProcess[str]:
    """
    Run a shell command with error handling and input validation.

    Args:
        cmd: Command and arguments as a list
        capture_output: Whether to capture stdout/stderr
        check: Whether to raise on non-zero exit code
        timeout: Command timeout in seconds

    Returns:
        CompletedProcess with command result

    Raises:
        DiagnosticError: For command execution errors
        ConfigurationError: For invalid input
    """
    from .validators import sanitize_command, validate_timeout

    # Validate and sanitize inputs
    cmd = sanitize_command(cmd)
    timeout = validate_timeout(timeout)

    try:
        result = subprocess.run(
            cmd,
            capture_output=capture_output,
            text=True,
            check=check,
            timeout=timeout,
        )
        return result
    except subprocess.TimeoutExpired as e:
        error_msg = f"Command timed out after {timeout}s: {' '.join(cmd)}"
        console.print(f"[red]{error_msg}[/red]")
        raise DiagnosticError(
            error_msg,
            "Try increasing timeout or check if the command is hanging",
        ) from e
    except FileNotFoundError as e:
        error_msg = f"Command not found: {cmd[0]}"
        console.print(f"[red]{error_msg}[/red]")
        raise DiagnosticError(
            error_msg,
            f"Install {cmd[0]} or ensure it's in your PATH",
        ) from e
    except subprocess.CalledProcessError as e:
        if not check:
            # Return the exception as if it were a result
            # This maintains backward compatibility but is type-unsafe
            return subprocess.CompletedProcess(  # type: ignore[return-value]
                cmd, e.returncode, e.stdout, e.stderr
            )
        raise


def check_command_exists(cmd: str) -> bool:
    """
    Check if a command exists in PATH.

    Args:
        cmd: Command name to check

    Returns:
        True if command exists and is executable, False otherwise
    """
    if not isinstance(cmd, str) or not cmd.strip():
        return False

    try:
        # Use 'which' on Unix, 'where' on Windows
        if sys.platform == "win32":
            check_cmd = ["where", cmd]
        else:
            check_cmd = ["which", cmd]

        result = subprocess.run(
            check_cmd,
            capture_output=True,
            timeout=5,
            check=False,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return False


def get_home_config_dir() -> Path:
    """Get the configuration directory for mlenvdoctor."""
    home = Path.home()
    config_dir = home / ".mlenvdoctor"
    config_dir.mkdir(exist_ok=True)
    return config_dir


def print_success(message: str) -> None:
    """Print a success message."""
    console.print(f"[green]{icon_check()} {message}[/green]")


def print_error(message: str) -> None:
    """Print an error message."""
    console.print(f"[red]{icon_cross()} {message}[/red]")


def print_warning(message: str) -> None:
    """Print a warning message."""
    console.print(f"[yellow]{icon_warning()}  {message}[/yellow]")


def print_info(message: str) -> None:
    """Print an info message."""
    console.print(f"[blue]{icon_info()}  {message}[/blue]")


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
