"""Export functionality for diagnostic results."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .diagnose import DiagnosticIssue


def issue_to_dict(issue: DiagnosticIssue) -> Dict[str, Any]:
    """Convert DiagnosticIssue to dictionary."""
    return {
        "name": issue.name,
        "status": issue.status,
        "severity": issue.severity,
        "fix": issue.fix,
        "details": issue.details,
    }


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

    # Convert issues to dictionaries
    issues_data = [issue_to_dict(issue) for issue in issues]

    # Calculate summary
    critical_count = sum(
        1 for i in issues if i.severity == "critical" and "FAIL" in i.status
    )
    warning_count = sum(
        1 for i in issues if i.severity == "warning" and ("WARN" in i.status or "FAIL" in i.status)
    )
    pass_count = sum(1 for i in issues if "PASS" in i.status)

    # Build export data
    export_data: Dict[str, Any] = {
        "issues": issues_data,
        "summary": {
            "total": len(issues),
            "passed": pass_count,
            "warnings": warning_count,
            "critical": critical_count,
        },
    }

    # Add metadata if requested
    if include_metadata:
        from . import __version__

        export_data["metadata"] = {
            "version": __version__,
            "timestamp": datetime.now().isoformat(),
            "tool": "mlenvdoctor",
        }

    # Write to file
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
        writer.writerow(["Issue", "Status", "Severity", "Fix", "Details"])

        for issue in issues:
            writer.writerow(
                [
                    issue.name,
                    issue.status,
                    issue.severity,
                    issue.fix,
                    issue.details or "",
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

    # Calculate summary
    critical_count = sum(
        1 for i in issues if i.severity == "critical" and "FAIL" in i.status
    )
    warning_count = sum(
        1 for i in issues if i.severity == "warning" and ("WARN" in i.status or "FAIL" in i.status)
    )
    pass_count = sum(1 for i in issues if "PASS" in i.status)

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
            <div class="value">{len(issues)}</div>
        </div>
        <div class="summary-card">
            <h3>Passed</h3>
            <div class="value passed">{pass_count}</div>
        </div>
        <div class="summary-card">
            <h3>Warnings</h3>
            <div class="value warning">{warning_count}</div>
        </div>
        <div class="summary-card">
            <h3>Critical Issues</h3>
            <div class="value critical">{critical_count}</div>
        </div>
    </div>

    <table>
        <thead>
            <tr>
                <th>Issue</th>
                <th>Status</th>
                <th>Severity</th>
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
                <td><strong>{issue.name}</strong></td>
                <td class="{status_class}">{issue.status}</td>
                <td>{issue.severity.upper()}</td>
                <td>{issue.fix or '-'}</td>
                <td>{issue.details or '-'}</td>
            </tr>
"""

    html_content += """
        </tbody>
    </table>

    <div class="footer">
        <p>Generated by ML Environment Doctor | <a href="https://github.com/dheena731/ml_env_doctor">GitHub</a></p>
    </div>
</body>
</html>
"""

    output_file.write_text(html_content, encoding="utf-8")
    return output_file
