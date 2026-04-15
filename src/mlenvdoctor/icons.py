"""Safe icon handling for cross-platform compatibility."""

from typing import Literal

_ICONS = {
    "search": "[SCAN]",
    "check": "[OK]",
    "cross": "[X]",
    "warning": "[!]",
    "info": "[i]",
    "wrench": "[FIX]",
    "whale": "[DOCKER]",
    "test": "[TEST]",
}


def get_icon(
    icon_name: Literal["search", "check", "cross", "warning", "info", "wrench", "whale", "test"],
) -> str:
    """Get a safe ASCII icon for the current platform."""
    return _ICONS.get(icon_name, "")


def icon_search() -> str:
    """Get search icon."""
    return get_icon("search")


def icon_check() -> str:
    """Get check icon."""
    return get_icon("check")


def icon_cross() -> str:
    """Get cross/error icon."""
    return get_icon("cross")


def icon_warning() -> str:
    """Get warning icon."""
    return get_icon("warning")


def icon_info() -> str:
    """Get info icon."""
    return get_icon("info")


def icon_wrench() -> str:
    """Get wrench/fix icon."""
    return get_icon("wrench")


def icon_whale() -> str:
    """Get whale/docker icon."""
    return get_icon("whale")


def icon_test() -> str:
    """Get test icon."""
    return get_icon("test")
