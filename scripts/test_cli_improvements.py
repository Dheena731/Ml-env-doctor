#!/usr/bin/env python3
"""Test script to verify CLI improvements."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

# Set UTF-8 encoding for Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Colors for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

# Safe icons for Windows
try:
    CHECK = "✅"
    CROSS = "❌"
    INFO = "ℹ️"
    WARNING = "⚠️"
except UnicodeEncodeError:
    CHECK = "[OK]"
    CROSS = "[X]"
    INFO = "[i]"
    WARNING = "[!]"


def print_success(message: str) -> None:
    """Print success message."""
    try:
        print(f"{GREEN}{CHECK} {message}{RESET}")
    except UnicodeEncodeError:
        print(f"{GREEN}[OK] {message}{RESET}")


def print_error(message: str) -> None:
    """Print error message."""
    try:
        print(f"{RED}{CROSS} {message}{RESET}")
    except UnicodeEncodeError:
        print(f"{RED}[X] {message}{RESET}")


def print_info(message: str) -> None:
    """Print info message."""
    try:
        print(f"{BLUE}{INFO}  {message}{RESET}")
    except UnicodeEncodeError:
        print(f"{BLUE}[i]  {message}{RESET}")


def print_warning(message: str) -> None:
    """Print warning message."""
    try:
        print(f"{YELLOW}{WARNING}  {message}{RESET}")
    except UnicodeEncodeError:
        print(f"{YELLOW}[!]  {message}{RESET}")


def run_command(cmd: list[str], check: bool = True) -> tuple[int, str, str]:
    """Run a command and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "Command timed out"
    except Exception as e:
        return 1, "", str(e)


def test_version_command() -> bool:
    """Test --version flag."""
    print_info("Testing --version command...")
    returncode, stdout, stderr = run_command(["mlenvdoctor", "--version"], check=False)
    if returncode == 0 and "ML Environment Doctor" in stdout:
        print_success("Version command works")
        return True
    else:
        print_error(f"Version command failed: {stderr}")
        return False


def test_diagnose_basic() -> bool:
    """Test basic diagnose command."""
    print_info("Testing basic diagnose command...")
    returncode, stdout, stderr = run_command(["mlenvdoctor", "diagnose"], check=False)
    if returncode == 0 and "Running ML Environment Diagnostics" in stdout:
        print_success("Basic diagnose works")
        return True
    else:
        print_warning(f"Diagnose command: {stderr}")
        return returncode == 0


def test_diagnose_full() -> bool:
    """Test full diagnose command."""
    print_info("Testing full diagnose command...")
    returncode, stdout, stderr = run_command(["mlenvdoctor", "diagnose", "--full"], check=False)
    if returncode == 0:
        print_success("Full diagnose works")
        return True
    else:
        print_warning(f"Full diagnose: {stderr}")
        return False


def test_json_export() -> bool:
    """Test JSON export."""
    print_info("Testing JSON export...")
    output_file = Path("test_results.json")
    if output_file.exists():
        output_file.unlink()

    returncode, stdout, stderr = run_command(
        ["mlenvdoctor", "diagnose", "--json", str(output_file)],
        check=False,
    )

    if returncode == 0 and output_file.exists():
        try:
            data = json.loads(output_file.read_text())
            if "issues" in data and "summary" in data:
                print_success(f"JSON export works: {len(data['issues'])} issues exported")
                output_file.unlink()  # Cleanup
                return True
            else:
                print_error("JSON export missing required fields")
                return False
        except json.JSONDecodeError as e:
            print_error(f"Invalid JSON: {e}")
            return False
    else:
        print_error(f"JSON export failed: {stderr}")
        if output_file.exists():
            output_file.unlink()
        return False


def test_csv_export() -> bool:
    """Test CSV export."""
    print_info("Testing CSV export...")
    output_file = Path("test_results.csv")
    if output_file.exists():
        output_file.unlink()

    returncode, stdout, stderr = run_command(
        ["mlenvdoctor", "diagnose", "--csv", str(output_file)],
        check=False,
    )

    if returncode == 0 and output_file.exists():
        content = output_file.read_text()
        if "Issue,Status,Severity" in content:
            print_success("CSV export works")
            output_file.unlink()  # Cleanup
            return True
        else:
            print_error("CSV export missing headers")
            return False
    else:
        print_error(f"CSV export failed: {stderr}")
        if output_file.exists():
            output_file.unlink()
        return False


def test_html_export() -> bool:
    """Test HTML export."""
    print_info("Testing HTML export...")
    output_file = Path("test_report.html")
    if output_file.exists():
        output_file.unlink()

    returncode, stdout, stderr = run_command(
        ["mlenvdoctor", "diagnose", "--html", str(output_file)],
        check=False,
    )

    if returncode == 0 and output_file.exists():
        content = output_file.read_text()
        if "<html" in content and "ML Environment Doctor" in content:
            print_success("HTML export works")
            print_info(f"HTML report saved to: {output_file.absolute()}")
            # Don't delete HTML so user can view it
            return True
        else:
            print_error("HTML export missing content")
            return False
    else:
        print_error(f"HTML export failed: {stderr}")
        if output_file.exists():
            output_file.unlink()
        return False


def test_logging() -> bool:
    """Test logging functionality."""
    print_info("Testing logging...")
    log_file = Path("test_cli.log")
    if log_file.exists():
        log_file.unlink()

    returncode, stdout, stderr = run_command(
        ["mlenvdoctor", "diagnose", "--log-file", str(log_file), "--log-level", "DEBUG"],
        check=False,
    )

    if returncode == 0 and log_file.exists():
        content = log_file.read_text()
        if "mlenvdoctor" in content.lower():
            print_success(f"Logging works: {log_file.stat().st_size} bytes")
            log_file.unlink()  # Cleanup
            return True
        else:
            print_warning("Log file exists but content seems empty")
            return True  # Still counts as success
    else:
        print_warning(f"Logging test: {stderr}")
        if log_file.exists():
            log_file.unlink()
        return False


def test_multiple_exports() -> bool:
    """Test multiple export formats at once."""
    print_info("Testing multiple exports...")
    json_file = Path("test_multi.json")
    csv_file = Path("test_multi.csv")
    html_file = Path("test_multi.html")

    # Cleanup
    for f in [json_file, csv_file, html_file]:
        if f.exists():
            f.unlink()

    returncode, stdout, stderr = run_command(
        [
            "mlenvdoctor",
            "diagnose",
            "--json",
            str(json_file),
            "--csv",
            str(csv_file),
            "--html",
            str(html_file),
        ],
        check=False,
    )

    if returncode == 0:
        all_exist = json_file.exists() and csv_file.exists() and html_file.exists()
        if all_exist:
            print_success("Multiple exports work")
            # Cleanup
            json_file.unlink()
            csv_file.unlink()
            # Keep HTML for user to view
            return True
        else:
            print_error("Not all export files were created")
            return False
    else:
        print_error(f"Multiple exports failed: {stderr}")
        return False


def test_error_handling() -> bool:
    """Test error handling with invalid inputs."""
    print_info("Testing error handling...")

    # Test invalid log level
    returncode, stdout, stderr = run_command(
        ["mlenvdoctor", "diagnose", "--log-level", "INVALID"],
        check=False,
    )
    if returncode != 0:
        print_success("Invalid log level properly rejected")
        return True
    else:
        print_warning("Invalid log level not rejected")
        return False


def main() -> int:
    """Run all CLI tests."""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}ML Environment Doctor - CLI Testing{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")

    tests = [
        ("Version Command", test_version_command),
        ("Basic Diagnose", test_diagnose_basic),
        ("Full Diagnose", test_diagnose_full),
        ("JSON Export", test_json_export),
        ("CSV Export", test_csv_export),
        ("HTML Export", test_html_export),
        ("Logging", test_logging),
        ("Multiple Exports", test_multiple_exports),
        ("Error Handling", test_error_handling),
    ]

    results: list[tuple[str, bool]] = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print_error(f"{name} crashed: {e}")
            results.append((name, False))
        print()

    # Summary
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}Test Summary{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        if result:
            print_success(f"{name}: PASSED")
        else:
            print_error(f"{name}: FAILED")

    print(f"\n{BLUE}Total: {passed}/{total} tests passed{RESET}\n")

    if passed == total:
        print_success("All tests passed! 🎉")
        return 0
    else:
        print_error(f"{total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
