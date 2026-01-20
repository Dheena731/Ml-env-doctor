"""Tests for diagnose module."""

import pytest

from mlenvdoctor.diagnose import DiagnosticIssue, diagnose_env


def test_diagnostic_issue():
    """Test DiagnosticIssue class."""
    issue = DiagnosticIssue(
        name="Test Issue",
        status="FAIL",
        severity="critical",
        fix="Fix it",
        details="Details",
    )
    assert issue.name == "Test Issue"
    assert issue.status == "FAIL"
    assert issue.severity == "critical"
    assert issue.fix == "Fix it"
    assert issue.details == "Details"

    row = issue.to_row()
    assert len(row) == 4
    assert row[0] == "Test Issue"


def test_diagnose_env():
    """Test diagnose_env function."""
    issues = diagnose_env(full=False)
    assert isinstance(issues, list)
    assert all(isinstance(issue, DiagnosticIssue) for issue in issues)

    # Should have at least some issues
    assert len(issues) > 0


def test_diagnose_env_full():
    """Test full diagnose_env."""
    issues = diagnose_env(full=True)
    assert isinstance(issues, list)
    assert all(isinstance(issue, DiagnosticIssue) for issue in issues)
    # Full scan should return more issues
    assert len(issues) >= 3  # At least basic checks
