"""Configuration management for ML Environment Doctor."""

from pathlib import Path
from typing import Any, Dict, Optional

# Try tomllib (Python 3.11+)
try:
    import tomllib
except ImportError:
    tomllib = None  # type: ignore

# Fallback to tomli for older Python versions
try:
    import tomli
except ImportError:
    tomli = None

from .exceptions import ConfigurationError
from .utils import get_home_config_dir


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load configuration from TOML file.

    Args:
        config_path: Path to config file. If None, searches for:
            1. mlenvdoctor.toml in current directory
            2. .mlenvdoctorrc in current directory
            3. ~/.mlenvdoctor/config.toml

    Returns:
        Configuration dictionary

    Raises:
        ConfigurationError: If config file is invalid
    """
    default_config: Dict[str, Any] = {
        "diagnostics": {
            "full_scan": False,
            "skip_checks": [],
        },
        "fix": {
            "default_stack": "trl-peft",
            "auto_install": False,
        },
        "docker": {
            "default_base_image": "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        },
        "logging": {
            "level": "INFO",
            "file": None,
        },
    }

    if config_path is None:
        # Search for config files
        search_paths = [
            Path("mlenvdoctor.toml"),
            Path(".mlenvdoctorrc"),
            get_home_config_dir() / "config.toml",
        ]

        for path in search_paths:
            if path.exists():
                config_path = path
                break
    else:
        if not config_path.exists():
            raise ConfigurationError(
                f"Config file not found: {config_path}",
                "Create the file or use default configuration",
            )

    if config_path is None or not config_path.exists():
        return default_config

    try:
        # Try tomllib (Python 3.11+)
        if tomllib is not None:
            with config_path.open("rb") as f:
                user_config = tomllib.load(f)
        elif tomli is not None:
            # Fallback to tomli for older Python
            with config_path.open("rb") as f:
                user_config = tomli.load(f)
        else:
            raise ConfigurationError(
                "TOML parsing not available. Install tomli: pip install tomli",
                "Or upgrade to Python 3.11+",
            )

        # Merge with defaults
        merged_config = default_config.copy()
        for section, values in user_config.items():
            if section in merged_config and isinstance(merged_config[section], dict):
                merged_config[section].update(values)
            else:
                merged_config[section] = values

        return merged_config

    except Exception as e:
        raise ConfigurationError(
            f"Error parsing config file {config_path}: {e}",
            "Check TOML syntax and file permissions",
        ) from e


def get_config_value(config: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """
    Get nested config value safely.

    Args:
        config: Configuration dictionary
        *keys: Nested keys to traverse
        default: Default value if key not found

    Returns:
        Config value or default
    """
    value = config
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
            if value is None:
                return default
        else:
            return default
    return value if value is not None else default


def create_default_config(output_path: Path) -> Path:
    """
    Create a default configuration file.

    Args:
        output_path: Path where to create config file

    Returns:
        Path to created config file
    """
    default_content = """# ML Environment Doctor Configuration

[diagnostics]
# Run full scan by default
full_scan = false
# Skip specific checks (e.g., ["docker_gpu", "internet"])
skip_checks = []

[fix]
# Default ML stack: "trl-peft" or "minimal"
default_stack = "trl-peft"
# Automatically install dependencies without prompting
auto_install = false

[docker]
# Default base image for Dockerfiles
default_base_image = "nvidia/cuda:12.4.0-devel-ubuntu22.04"

[logging]
# Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
level = "INFO"
# Log file path (None for default: ~/.mlenvdoctor/logs/mlenvdoctor.log)
file = null
"""

    output_path.write_text(default_content, encoding="utf-8")
    return output_path
