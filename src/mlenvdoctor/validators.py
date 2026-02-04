"""Input validation and sanitization for ML Environment Doctor."""

import re
from pathlib import Path
from typing import Optional

from .exceptions import ConfigurationError


def validate_model_name(model_name: str) -> str:
    """
    Validate and sanitize model name.

    Args:
        model_name: Model name to validate

    Returns:
        Sanitized model name

    Raises:
        ConfigurationError: If model name is invalid
    """
    if not model_name or not isinstance(model_name, str):
        raise ConfigurationError(
            "Model name must be a non-empty string",
            "Use a valid model name like 'tinyllama', 'gpt2', or 'mistral-7b'",
        )

    # Remove whitespace
    model_name = model_name.strip()

    # Check for dangerous characters (basic sanitization)
    if not re.match(r"^[a-zA-Z0-9._-]+$", model_name):
        raise ConfigurationError(
            f"Invalid model name: {model_name}",
            "Model name can only contain letters, numbers, dots, underscores, and hyphens",
        )

    return model_name.lower()


def validate_file_path(file_path: Path, must_exist: bool = False, must_be_file: bool = False) -> Path:
    """
    Validate and sanitize file path.

    Args:
        file_path: Path to validate
        must_exist: Whether the path must exist
        must_be_file: Whether the path must be a file

    Returns:
        Resolved, absolute path

    Raises:
        ConfigurationError: If path is invalid
    """
    if not isinstance(file_path, (Path, str)):
        raise ConfigurationError(
            "File path must be a Path object or string",
            "Use pathlib.Path or a valid string path",
        )

    path = Path(file_path).resolve()

    # Check for path traversal attempts
    if ".." in str(path):
        # Resolve should handle this, but double-check
        resolved = path.resolve()
        if ".." in str(resolved):
            raise ConfigurationError(
                "Invalid path: contains '..'",
                "Use absolute paths or relative paths without '..'",
            )

    if must_exist and not path.exists():
        raise ConfigurationError(
            f"Path does not exist: {path}",
            "Ensure the file or directory exists",
        )

    if must_be_file and not path.is_file():
        raise ConfigurationError(
            f"Path is not a file: {path}",
            "Provide a valid file path",
        )

    return path


def validate_log_level(level: str) -> str:
    """
    Validate logging level.

    Args:
        level: Log level to validate

    Returns:
        Validated log level

    Raises:
        ConfigurationError: If level is invalid
    """
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    level_upper = level.upper() if isinstance(level, str) else str(level).upper()

    if level_upper not in valid_levels:
        raise ConfigurationError(
            f"Invalid log level: {level}",
            f"Use one of: {', '.join(valid_levels)}",
        )

    return level_upper


def validate_stack_name(stack: str) -> str:
    """
    Validate ML stack name.

    Args:
        stack: Stack name to validate

    Returns:
        Validated stack name

    Raises:
        ConfigurationError: If stack is invalid
    """
    valid_stacks = {"trl-peft", "minimal"}
    stack_lower = stack.lower() if isinstance(stack, str) else str(stack).lower()

    if stack_lower not in valid_stacks:
        raise ConfigurationError(
            f"Invalid stack: {stack}",
            f"Use one of: {', '.join(valid_stacks)}",
        )

    return stack_lower


def sanitize_command(cmd: list[str]) -> list[str]:
    """
    Sanitize command arguments to prevent injection.

    Args:
        cmd: Command and arguments list

    Returns:
        Sanitized command list

    Raises:
        ConfigurationError: If command contains dangerous patterns
    """
    if not isinstance(cmd, list) or not cmd:
        raise ConfigurationError(
            "Command must be a non-empty list",
            "Provide command as a list of strings",
        )

    sanitized = []
    for arg in cmd:
        if not isinstance(arg, str):
            raise ConfigurationError(
                "All command arguments must be strings",
                "Convert all arguments to strings",
            )

        # Check for command injection patterns
        dangerous_patterns = [";", "&&", "||", "`", "$(", "<", ">", "|"]
        for pattern in dangerous_patterns:
            if pattern in arg:
                raise ConfigurationError(
                    f"Dangerous pattern detected in command: {pattern}",
                    "Do not use shell operators in command arguments",
                )

        sanitized.append(arg)

    return sanitized


def validate_timeout(timeout: Optional[int], min_timeout: int = 1, max_timeout: int = 3600) -> Optional[int]:
    """
    Validate timeout value.

    Args:
        timeout: Timeout in seconds
        min_timeout: Minimum allowed timeout
        max_timeout: Maximum allowed timeout

    Returns:
        Validated timeout

    Raises:
        ConfigurationError: If timeout is invalid
    """
    if timeout is None:
        return None

    if not isinstance(timeout, int):
        raise ConfigurationError(
            f"Timeout must be an integer, got {type(timeout)}",
            "Provide timeout as an integer number of seconds",
        )

    if timeout < min_timeout:
        raise ConfigurationError(
            f"Timeout too small: {timeout}s (minimum: {min_timeout}s)",
            f"Increase timeout to at least {min_timeout} seconds",
        )

    if timeout > max_timeout:
        raise ConfigurationError(
            f"Timeout too large: {timeout}s (maximum: {max_timeout}s)",
            f"Decrease timeout to at most {max_timeout} seconds",
        )

    return timeout
