"""Logging configuration for ML Environment Doctor."""

import logging
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler

from .utils import get_home_config_dir

console = Console()


def setup_logger(
    name: str = "mlenvdoctor",
    level: str = "INFO",
    log_file: Optional[Path] = None,
    enable_rich: bool = True,
) -> logging.Logger:
    """
    Set up logger with Rich console handler and optional file handler.

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        enable_rich: Use Rich handler for console output

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Console handler with Rich formatting
    if enable_rich:
        console_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
            tracebacks_show_locals=False,
        )
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

    console_handler.setLevel(getattr(logging, level.upper()))
    logger.addHandler(console_handler)

    # File handler if log file specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
            )
        )
        logger.addHandler(file_handler)

    return logger


def get_default_log_file() -> Path:
    """Get default log file path."""
    log_dir = get_home_config_dir() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "mlenvdoctor.log"


# Default logger instance
logger = setup_logger()
