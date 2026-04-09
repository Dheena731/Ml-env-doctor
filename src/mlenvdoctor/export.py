"""Export functionality for diagnostic results."""

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .diagnose import DiagnosticIssue, get_fix_commands


def issue_to_dict(issue: DiagnosticIssue) -> Dict[str, Any]:
    """Convert DiagnosticIssue to dictionary."""
    return {
        "name": issue.name,
        "status": issue.status,
        "severity": issue.severity,
        "fix": issue.fix,
        "details": issue.details,
        "check_id": issue.check_id,
        "category": issue.category,
        "recommendation": issue.recommendation,
        "likely_cause": issue.likely_cause,
        "verify_steps": issue.verify_steps,
        "confidence": issue.confidence,
        "evidence": issue.evidence,
        "metadata": issue.metadata,
    }


def build_summary(issues: List[DiagnosticIssue]) -> Dict[str, int]:
    """Build a diagnostic summary block."""
    critical_count = sum(1 for i in issues if i.severity == "critical" and "FAIL" in i.status)
    warning_count = sum(
        1 for i in issues if i.severity == "warning" and ("WARN" in i.status or "FAIL" in i.status)
    )
    pass_count = sum(1 for i in issues if "PASS" in i.status)
    return {
        "total": len(issues),
        "passed": pass_count,
        "warnings": warning_count,
        "critical": critical_count,
    }


def get_exit_code(issues: List[DiagnosticIssue]) -> int:
    """Return a machine-readable exit code for diagnostics."""
    summary = build_summary(issues)
    if summary["critical"] > 0:
        return 2
    if summary["warnings"] > 0:
        return 1
    return 0


def build_export_data(
    issues: List[DiagnosticIssue],
    include_metadata: bool = True,
) -> Dict[str, Any]:
    """Build the shared machine-readable diagnostics payload."""
    export_data: Dict[str, Any] = {
        "issues": [issue_to_dict(issue) for issue in issues],
        "summary": build_summary(issues),
        "exit_code": get_exit_code(issues),
        "fixes": get_fix_commands(issues),
    }

    if include_metadata:
        from . import __version__

        export_data["metadata"] = {
            "version": __version__,
            "timestamp": datetime.now().isoformat(),
            "tool": "mlenvdoctor",
        }

    return export_data


def export_json(
    issues: List[DiagnosticIssue],
    output_file: Optional[Path] = None,
    include_metadata: bool = True,
) -> Path:
    """
    Export diagnostic results to JSON.

    Args:
        issues: List of diagnostic issues
        output_file: Output file path (default: diagnostic-results.json)
        include_metadata: Include metadata (timestamp, version, etc.)

    Returns:
        Path to exported file
    """
    if output_file is None:
        output_file = Path("diagnostic-results.json")

    export_data = build_export_data(issues, include_metadata=include_metadata)
    output_file.write_text(json.dumps(export_data, indent=2, ensure_ascii=False), encoding="utf-8")

    return output_file


def export_csv(issues: List[DiagnosticIssue], output_file: Optional[Path] = None) -> Path:
    """
    Export diagnostic results to CSV.

    Args:
        issues: List of diagnostic issues
        output_file: Output file path (default: diagnostic-results.csv)

    Returns:
        Path to exported file
    """
    import csv

    if output_file is None:
        output_file = Path("diagnostic-results.csv")

    with output_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Issue",
                "Status",
                "Severity",
                "Category",
                "Check ID",
                "Fix",
                "Details",
                "Confidence",
            ]
        )

        for issue in issues:
            writer.writerow(
                [
                    issue.name,
                    issue.status,
                    issue.severity,
                    issue.category,
                    issue.check_id,
                    issue.fix,
                    issue.details or "",
                    issue.confidence,
                ]
            )

    return output_file


def export_html(issues: List[DiagnosticIssue], output_file: Optional[Path] = None) -> Path:
    """
    Export diagnostic results to HTML report.

    Args:
        issues: List of diagnostic issues
        output_file: Output file path (default: diagnostic-results.html)

    Returns:
        Path to exported file
    """
    if output_file is None:
        output_file = Path("diagnostic-results.html")

    summary = build_summary(issues)

    from . import __version__

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML Environment Doctor - Diagnostic Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        .summary-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .summary-card h3 {{
            margin: 0 0 10px 0;
            font-size: 14px;
            color: #666;
        }}
        .summary-card .value {{
            font-size: 32px;
            font-weight: bold;
        }}
        .passed {{ color: #10b981; }}
        .warning {{ color: #f59e0b; }}
        .critical {{ color: #ef4444; }}
        table {{
            width: 100%;
            background: white;
            border-collapse: collapse;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th {{
            background-color: #667eea;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #e5e7eb;
        }}
        tr:hover {{
            background-color: #f9fafb;
        }}
        .status-pass {{ color: #10b981; font-weight: 600; }}
        .status-fail {{ color: #ef4444; font-weight: 600; }}
        .status-warn {{ color: #f59e0b; font-weight: 600; }}
        .severity-critical {{ background-color: #fee2e2; }}
        .severity-warning {{ background-color: #fef3c7; }}
        .severity-info {{ background-color: #dbeafe; }}
        .footer {{
            margin-top: 30px;
            text-align: center;
            color: #666;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>ML Environment Doctor</h1>
        <p>Diagnostic Report - Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p style="font-size: 14px; opacity: 0.9;">Version {__version__}</p>
    </div>

    <div class="summary">
        <div class="summary-card">
            <h3>Total Checks</h3>
            <div class="value">{summary["total"]}</div>
        </div>
        <div class="summary-card">
            <h3>Passed</h3>
            <div class="value passed">{summary["passed"]}</div>
        </div>
        <div class="summary-card">
            <h3>Warnings</h3>
            <div class="value warning">{summary["warnings"]}</div>
        </div>
        <div class="summary-card">
            <h3>Critical Issues</h3>
            <div class="value critical">{summary["critical"]}</div>
        </div>
    </div>

    <table>
        <thead>
            <tr>
                <th>Issue</th>
                <th>Status</th>
                <th>Severity</th>
                <th>Category</th>
                <th>Fix</th>
                <th>Details</th>
            </tr>
        </thead>
        <tbody>
"""

    for issue in issues:
        status_class = "status-pass"
        if "FAIL" in issue.status:
            status_class = "status-fail"
        elif "WARN" in issue.status:
            status_class = "status-warn"

        severity_class = f"severity-{issue.severity}"

        html_content += f"""
            <tr class="{severity_class}">
                <td><strong>{html.escape(issue.name)}</strong></td>
                <td class="{status_class}">{html.escape(issue.status)}</td>
                <td>{html.escape(issue.severity.upper())}</td>
                <td>{html.escape(issue.category or '-')}</td>
                <td>{html.escape(issue.fix or '-')}</td>
                <td>{html.escape(issue.details or '-')}</td>
            </tr>
"""

    html_content += """
        </tbody>
    </table>

    <div class="footer">
        <p>Generated by ML Environment Doctor | <a href="https://github.com/Dheena731/Ml-env-doctor">GitHub</a></p>
    </div>
</body>
</html>
"""

    output_file.write_text(html_content, encoding="utf-8")
    return output_file
