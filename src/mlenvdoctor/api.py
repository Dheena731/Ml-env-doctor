"""Public Python API for ML Environment Doctor."""

from __future__ import annotations

from typing import List

from .diagnose import DiagnosticIssue, diagnose_env


def diagnose(*, full: bool = False, parallel: bool = True) -> List[DiagnosticIssue]:
    """Run ML environment diagnostics from Python code.

    Args:
        full: Include slower checks such as disk, Docker GPU, and network checks.
        parallel: Run independent checks concurrently when possible.

    Returns:
        A list of normalized diagnostic findings.
    """
    return diagnose_env(full=full, parallel=parallel, show_header=False)
