"""Minimal MCP-style server stub for ML Environment Doctor."""

import json
import sys
from typing import Any, Dict

from .diagnose import diagnose_env, get_fix_commands
from .export import build_export_data


def _handle_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Handle a single MCP stub request payload."""
    tool_name = payload.get("tool")
    arguments = payload.get("arguments", {})

    if tool_name == "diagnose":
        issues = diagnose_env(full=bool(arguments.get("full", False)), show_header=False)
        return {
            "ok": True,
            "tool": "diagnose",
            "result": build_export_data(issues, include_metadata=True),
        }

    if tool_name == "get_fixes":
        issues = diagnose_env(full=bool(arguments.get("full", False)), show_header=False)
        return {
            "ok": True,
            "tool": "get_fixes",
            "result": {
                "fixes": get_fix_commands(issues),
                "exit_code": build_export_data(issues, include_metadata=False)["exit_code"],
            },
        }

    return {
        "ok": False,
        "error": f"Unknown tool: {tool_name}",
        "available_tools": ["diagnose", "get_fixes"],
    }


def serve_mcp() -> int:
    """Serve a minimal JSON-line MCP stub over stdin/stdout."""
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            sys.stdout.write(
                json.dumps({"ok": False, "error": f"Invalid JSON request: {exc}"}) + "\n"
            )
            sys.stdout.flush()
            continue

        response = _handle_request(request)
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    return 0
