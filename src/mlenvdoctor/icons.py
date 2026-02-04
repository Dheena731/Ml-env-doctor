"""Safe emoji/icon handling for cross-platform compatibility."""

import sys
from typing import Literal

# Check if we can safely use emojis
_USE_EMOJIS = True

try:
    # Try to write an emoji to see if it works
    if sys.platform == "win32":
        import io

        # Test if console supports UTF-8
        test_output = io.StringIO()
        try:
            test_output.write("🔍")
            test_output.getvalue()
        except (UnicodeEncodeError, UnicodeError):
            _USE_EMOJIS = False
except Exception:
    _USE_EMOJIS = False


def get_icon(icon_name: Literal["search", "check", "cross", "warning", "info", "wrench", "whale", "test"]) -> str:
    """
    Get a safe icon/emoji for the current platform.

    Args:
        icon_name: Name of the icon to get

    Returns:
        Emoji if supported, ASCII alternative otherwise
    """
    if _USE_EMOJIS:
        icons = {
            "search": "🔍",
            "check": "✅",
            "cross": "❌",
            "warning": "⚠️",
            "info": "ℹ️",
            "wrench": "🔧",
            "whale": "🐳",
            "test": "🧪",
        }
    else:
        # ASCII alternatives
        icons = {
            "search": "[*]",
            "check": "[OK]",
            "cross": "[X]",
            "warning": "[!]",
            "info": "[i]",
            "wrench": "[FIX]",
            "whale": "[DOCKER]",
            "test": "[TEST]",
        }

    return icons.get(icon_name, "")


# Convenience functions
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
