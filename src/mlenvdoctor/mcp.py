"""Minimal JSON-lines MCP server for ML Environment Doctor."""

import json
import sys
from typing import Any, Dict

from . import __version__
from .diagnose import diagnose_env, get_fix_commands
from .export import DOCTOR_SUMMARY_SCHEMA_VERSION, build_export_data

MCP_SCHEMA_VERSION = "0.2"
MCP_CONTRACT_VERSION = "1.0"
MCP_COMPAT_POLICY = "additive-only-with-aliases"

MCP_TOOLS: Dict[str, Dict[str, Any]] = {
    "health": {
        "description": "Check MCP server availability and version.",
        "arguments": {},
    },
    "list_tools": {
        "description": "List available MCP tools and basic metadata.",
        "arguments": {},
    },
    "tool_schema": {
        "description": "Return argument schema and usage for a single tool.",
        "arguments": {"name": "str (required)"},
    },
    "diagnose": {
        "description": "Run diagnostics and return full machine-readable report.",
        "arguments": {"full": "bool (optional, default=false)"},
    },
    "get_fixes": {
        "description": "Run diagnostics and return suggested fix commands.",
        "arguments": {"full": "bool (optional, default=false)"},
    },
    "doctor_summary": {
        "description": "Run diagnostics and return doctor-oriented findings.",
        "arguments": {"full": "bool (optional, default=false)"},
    },
    "report_bundle": {
        "description": "Return a compact integration payload for assistant use.",
        "arguments": {
            "full": "bool (optional, default=false)",
            "include_metadata": "bool (optional, default=true)",
        },
    },
    "diagnose_environment": {
        "description": "MCP v1 alias for diagnose.",
        "arguments": {"full": "bool (optional, default=false)"},
    },
    "get_fix_plan": {
        "description": "MCP v1 alias for get_fixes.",
        "arguments": {"full": "bool (optional, default=false)"},
    },
    "verify_fix": {
        "description": "Run verification diagnostics and return doctor-style findings.",
        "arguments": {"full": "bool (optional, default=false)"},
    },
    "export_report": {
        "description": "MCP v1 alias for report_bundle.",
        "arguments": {
            "full": "bool (optional, default=false)",
            "include_metadata": "bool (optional, default=true)",
        },
    },
}

MCP_TOOL_ALIASES: Dict[str, str] = {
    "diagnose_environment": "diagnose",
    "get_fix_plan": "get_fixes",
    "verify_fix": "doctor_summary",
    "export_report": "report_bundle",
}


def _available_tools() -> list[str]:
    """Return stable, sorted list of available MCP tools."""
    return sorted(MCP_TOOLS.keys())


def _unknown_tool_response(tool_name: Any) -> Dict[str, Any]:
    """Build a stable unknown-tool error response."""
    return {
        "ok": False,
        "error": f"Unknown tool: {tool_name}",
        "available_tools": _available_tools(),
    }


def _handle_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Handle a single MCP request payload."""
    tool_name = payload.get("tool")
    arguments = payload.get("arguments", {})
    canonical_tool = MCP_TOOL_ALIASES.get(str(tool_name), tool_name)
    full = bool(arguments.get("full", False))

    if tool_name == "health":
        return {
            "ok": True,
            "tool": "health",
            "result": {
                "status": "ok",
                "mcp_schema_version": MCP_SCHEMA_VERSION,
                "mcp_contract_version": MCP_CONTRACT_VERSION,
                "compatibility_policy": MCP_COMPAT_POLICY,
                "tool": "mlenvdoctor",
                "version": __version__,
            },
        }

    if tool_name == "list_tools":
        return {
            "ok": True,
            "tool": "list_tools",
            "result": {
                "mcp_schema_version": MCP_SCHEMA_VERSION,
                "mcp_contract_version": MCP_CONTRACT_VERSION,
                "tools": [
                    {
                        "name": name,
                        "description": MCP_TOOLS[name]["description"],
                        "alias_for": MCP_TOOL_ALIASES.get(name),
                    }
                    for name in _available_tools()
                ],
            },
        }

    if tool_name == "tool_schema":
        target_tool = arguments.get("name")
        if not isinstance(target_tool, str) or not target_tool.strip():
            return {
                "ok": False,
                "error": "tool_schema requires a non-empty string argument: name",
                "available_tools": _available_tools(),
            }
        if target_tool not in MCP_TOOLS:
            return _unknown_tool_response(target_tool)
        target_canonical = MCP_TOOL_ALIASES.get(target_tool, target_tool)
        return {
            "ok": True,
            "tool": "tool_schema",
            "result": {
                "mcp_schema_version": MCP_SCHEMA_VERSION,
                "mcp_contract_version": MCP_CONTRACT_VERSION,
                "name": target_tool,
                "canonical_name": target_canonical,
                "description": MCP_TOOLS[target_tool]["description"],
                "arguments": MCP_TOOLS[target_tool]["arguments"],
            },
        }

    if canonical_tool == "diagnose":
        issues = diagnose_env(full=full, show_header=False)
        return {
            "ok": True,
            "tool": str(tool_name),
            "canonical_tool": "diagnose",
            "result": build_export_data(issues, include_metadata=True),
        }

    if canonical_tool == "get_fixes":
        issues = diagnose_env(full=full, show_header=False)
        export_data = build_export_data(issues, include_metadata=False)
        return {
            "ok": True,
            "tool": str(tool_name),
            "canonical_tool": "get_fixes",
            "result": {
                "fixes": get_fix_commands(issues),
                "exit_code": export_data["exit_code"],
                "summary": export_data["summary"],
            },
        }

    if canonical_tool == "doctor_summary":
        issues = diagnose_env(full=full, show_header=False)
        export_data = build_export_data(issues, include_metadata=True)
        return {
            "ok": True,
            "tool": str(tool_name),
            "canonical_tool": "doctor_summary",
            "result": {
                "schema_version": DOCTOR_SUMMARY_SCHEMA_VERSION,
                "doctor_summary": export_data["doctor_summary"],
                "runtime_context": export_data["runtime_context"],
                "exit_code": build_export_data(issues, include_metadata=False)["exit_code"],
            },
        }

    if canonical_tool == "report_bundle":
        include_metadata = bool(arguments.get("include_metadata", True))
        issues = diagnose_env(full=full, show_header=False)
        export_data = build_export_data(issues, include_metadata=include_metadata)
        return {
            "ok": True,
            "tool": str(tool_name),
            "canonical_tool": "report_bundle",
            "result": {
                "mcp_schema_version": MCP_SCHEMA_VERSION,
                "mcp_contract_version": MCP_CONTRACT_VERSION,
                "summary": export_data["summary"],
                "runtime_context": export_data["runtime_context"],
                "doctor_summary": export_data["doctor_summary"],
                "fixes": export_data["fixes"],
                "exit_code": export_data["exit_code"],
                "metadata": export_data.get("metadata"),
            },
        }

    return _unknown_tool_response(tool_name)


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
