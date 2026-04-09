"""CLI entrypoint for ML Environment Doctor."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
import typer
from typer.core import TyperArgument

from . import __version__
from .diagnose import diagnose_env, get_fix_commands, print_diagnostic_table, summarize_for_doctor
from .dockerize import generate_dockerfile, generate_service_template
from .export import build_export_data, export_csv, export_html, export_json, get_exit_code
from .fix import auto_fix, get_stack_requirements
from .mcp import serve_mcp
from .gpu import benchmark_gpu_ops, smoke_test_lora, test_model as gpu_test_model
from .icons import (
    icon_check,
    icon_cross,
    icon_search,
    icon_test,
    icon_warning,
    icon_whale,
    icon_wrench,
)
from .logger import get_default_log_file, setup_logger
from .utils import console
from .validators import validate_log_level


def _patch_typer_click_metavar_compat() -> None:
    """Patch Typer/Click metavar incompatibilities in older Typer releases."""
    if getattr(click.core.Parameter.make_metavar, "__mlenvdoctor_compat__", False):
        return

    original_parameter_make_metavar = click.core.Parameter.make_metavar

    def parameter_make_metavar(self, ctx=None):  # type: ignore[no-untyped-def]
        if ctx is None:
            ctx = click.Context(click.Command(self.name or "mlenvdoctor"))
        return original_parameter_make_metavar(self, ctx)

    parameter_make_metavar.__mlenvdoctor_compat__ = True  # type: ignore[attr-defined]
    click.core.Parameter.make_metavar = parameter_make_metavar

    def typer_argument_make_metavar(self, ctx=None):  # type: ignore[no-untyped-def]
        if ctx is None:
            ctx = click.Context(click.Command(self.name or "mlenvdoctor"))
        return click.core.Argument.make_metavar(self, ctx)

    TyperArgument.make_metavar = typer_argument_make_metavar


_patch_typer_click_metavar_compat()

app = typer.Typer(
    name="mlenvdoctor",
    help=f"{icon_search()} ML Environment Doctor - Diagnose & fix ML environments for LLM fine-tuning",
    add_completion=False,
)
mcp_app = typer.Typer(help="Minimal MCP server support.")
stack_app = typer.Typer(help="Recommended dependency stacks.")


def _print_doctor_summary(issues) -> None:
    """Print prioritized, non-overlapping `doctor` output."""
    findings = summarize_for_doctor(issues)
    if not findings:
        console.print(f"[bold green]{icon_check()} No actionable problems detected.[/bold green]")
        console.print("[green]Your next step: proceed, or run `mlenvdoctor diagnose` for full evidence.[/green]")
        return

    console.print(f"[bold blue]{icon_search()} Doctor Summary[/bold blue]\n")
    for index, finding in enumerate(findings, start=1):
        severity_color = "red" if finding.severity == "critical" else "yellow"
        console.print(f"[bold {severity_color}]{index}. Problem: {finding.problem}[/bold {severity_color}]")
        console.print(f"Likely cause: {finding.likely_cause}")
        console.print(f"Best next fix: {finding.best_fix}")
        console.print("Verify:")
        for step in finding.verify_steps:
            console.print(f"  [cyan]{step}[/cyan]")
        if finding.evidence:
            console.print(f"Evidence: {finding.evidence[0]}")
        console.print()


def version_callback(value: bool):
    """Print version and exit."""
    if value:
        console.print(
            f"[bold blue]ML Environment Doctor[/bold blue] version [cyan]{__version__}[/cyan]"
        )
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit",
    ),
    log_file: Optional[Path] = typer.Option(
        None,
        "--log-file",
        help="Path to log file (default: ~/.mlenvdoctor/logs/mlenvdoctor.log)",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL",
    ),
):
    """ML Environment Doctor - Diagnose & fix ML environments for LLM fine-tuning."""
    try:
        validated_level = validate_log_level(log_level)
    except Exception as exc:
        raise typer.BadParameter(str(exc), param_hint="--log-level") from exc

    try:
        log_path = log_file or get_default_log_file()
        setup_logger(log_file=log_path, level=validated_level)
    except OSError as exc:
        setup_logger(log_file=None, level=validated_level)
        typer.echo(f"{icon_warning()} Logging to file disabled: {exc}", err=True)


@app.command()
def diagnose(
    full: bool = typer.Option(
        False, "--full", "-f", help="Run full diagnostics including GPU benchmarks"
    ),
    json_output: Optional[str] = typer.Option(
        None,
        "--json",
        help="Export results to JSON file, or use '-' to print machine-readable JSON to stdout",
    ),
    csv_output: Optional[Path] = typer.Option(
        None, "--csv", help="Export results to CSV file"
    ),
    html_output: Optional[Path] = typer.Option(
        None, "--html", help="Export results to HTML file"
    ),
):
    f"""
    {icon_search()} Diagnose your environment with detailed evidence.

    Use this command when you want the full list of checks, detailed findings,
    and exportable report data. Full scan (--full) adds extended environment checks
    and benchmark output.
    """
    json_to_stdout = json_output == "-"
    issues = diagnose_env(full=full, show_header=not json_to_stdout)
    if not json_to_stdout:
        print_diagnostic_table(issues)
    exit_code = get_exit_code(issues)

    # Export to formats if requested
    if json_output:
        if json_to_stdout:
            typer.echo(json.dumps(build_export_data(issues, include_metadata=True), ensure_ascii=False))
        else:
            export_json(issues, Path(json_output))
            console.print(f"[green]{icon_check()} Exported to {json_output}[/green]")
    if csv_output:
        export_csv(issues, csv_output)
        console.print(f"[green]{icon_check()} Exported to {csv_output}[/green]")
    if html_output:
        export_html(issues, html_output)
        console.print(f"[green]{icon_check()} Exported to {html_output}[/green]")

    if full and not json_to_stdout:
        console.print()
        console.print("[bold blue]Running GPU benchmark...[/bold blue]")
        try:
            benchmarks = benchmark_gpu_ops()
            if benchmarks:
                console.print("[green]GPU benchmark results:[/green]")
                for op, time_ms in benchmarks.items():
                    console.print(f"  {op}: {time_ms:.2f} ms")
            else:
                console.print("[yellow]GPU benchmark skipped (no GPU available)[/yellow]")
        except Exception as e:
            console.print(f"[yellow]GPU benchmark error: {e}[/yellow]")

    raise typer.Exit(exit_code)


@app.command()
def doctor(
    ci: bool = typer.Option(False, "--ci", help="Emit CI-friendly output"),
    full: bool = typer.Option(
        False, "--full", "-f", help="Run full diagnostics including GPU benchmarks"
    ),
):
    f"""
    {icon_search()} Triage the environment and recommend the best next fix.

    Use this command for a compact, opinionated summary:
    what failed, the likely cause, the best next step, and how to verify it.
    """
    issues = diagnose_env(full=full, show_header=not ci)
    exit_code = get_exit_code(issues)
    findings = summarize_for_doctor(issues)

    if ci:
        payload = build_export_data(issues, include_metadata=True)
        summary = payload["summary"]
        typer.echo(
            f"mlenvdoctor status={exit_code} passed={summary['passed']} "
            f"warnings={summary['warnings']} critical={summary['critical']}"
        )
        for finding in findings:
            verify = finding.verify_steps[0] if finding.verify_steps else "mlenvdoctor diagnose"
            typer.echo(
                f"ISSUE {finding.problem} | cause={finding.likely_cause} | "
                f"fix={finding.best_fix} | verify={verify}"
            )
    else:
        _print_doctor_summary(issues)

    raise typer.Exit(exit_code)


@app.command()
def report(
    full: bool = typer.Option(
        True, "--full/--quick", help="Generate a full report or a quick report"
    ),
    output_dir: Path = typer.Option(
        Path("mlenvdoctor-report"),
        "--output-dir",
        "-o",
        help="Directory where the report files will be written",
    ),
):
    f"""
    {icon_search()} Save a shareable diagnostics report.

    Writes JSON and HTML reports that can be pasted into CI artifacts or shared with teammates.
    """
    issues = diagnose_env(full=full)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = output_dir / f"diagnostic-report-{timestamp}.json"
    html_path = output_dir / f"diagnostic-report-{timestamp}.html"

    export_json(issues, json_path)
    export_html(issues, html_path)

    console.print()
    console.print(f"[green]{icon_check()} Saved JSON report: {json_path}[/green]")
    console.print(f"[green]{icon_check()} Saved HTML report: {html_path}[/green]")

    fixes = get_fix_commands(issues)
    if fixes:
        console.print()
        console.print("[bold]Suggested fix commands:[/bold]")
        for fix in fixes:
            console.print(f"[yellow]- {fix['issue']}[/yellow]")
            console.print(f"  [cyan]{fix['command']}[/cyan]")

    raise typer.Exit(get_exit_code(issues))


@app.command()
def fix(
    conda: bool = typer.Option(False, "--conda", "-c", help="Generate conda environment file"),
    venv: bool = typer.Option(False, "--venv", "-v", help="Create virtual environment"),
    stack: str = typer.Option(
        "trl-peft",
        "--stack",
        "-s",
        help="ML stack: trl-peft, minimal, or llm-training",
    ),
    apply: bool = typer.Option(
        False,
        "--apply",
        help="Apply available automated actions and install requirements when applicable",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompts and proceed with the selected fix actions",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show the planned fix actions without writing files or installing packages",
    ),
):
    f"""
    {icon_wrench()} Auto-fix environment issues and generate requirements.

    Generates requirements.txt or conda environment file based on detected issues.
    Optionally creates a virtual environment and installs dependencies.
    """
    result = auto_fix(
        use_conda=conda,
        create_venv=venv,
        stack=stack,
        apply=apply,
        yes=yes,
        dry_run=dry_run,
    )
    if result.success:
        console.print()
        console.print(f"[bold green]{icon_check()} Auto-fix completed![/bold green]")
        if result.created_paths:
            for path in result.created_paths:
                console.print(f"[cyan]Created: {path}[/cyan]")
        console.print("[yellow]Run 'mlenvdoctor diagnose' to verify fixes[/yellow]")
    else:
        raise typer.Exit(1)


@stack_app.command("llm-training")
def stack_llm_training(
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Optional file path to write the stack requirements to",
    ),
):
    """Print the recommended dependency stack for LLM fine-tuning."""
    requirements = get_stack_requirements("llm-training")
    content = "\n".join(requirements) + "\n"

    if output is not None:
        output.write_text(content, encoding="utf-8")
        console.print(f"[green]{icon_check()} Wrote stack to {output}[/green]")
    else:
        typer.echo(content.rstrip())


@app.command()
def dockerize(
    model: Optional[str] = typer.Argument(None, help="Model name (mistral-7b, tinyllama, gpt2)"),
    service: bool = typer.Option(
        False, "--service", "-s", help="Generate FastAPI service template"
    ),
    stack: Optional[str] = typer.Option(
        None,
        "--stack",
        help="Dependency stack to bake into the image (defaults to model-specific stack)",
    ),
    base_image: Optional[str] = typer.Option(
        None,
        "--base-image",
        help="Override the CUDA base image used by the Dockerfile",
    ),
    python_version: str = typer.Option(
        "3.10",
        "--python-version",
        help="Python version package to install in the container, e.g. 3.10",
    ),
    output: str = typer.Option(
        "Dockerfile.mlenvdoctor", "--output", "-o", help="Output Dockerfile name"
    ),
):
    f"""
    {icon_whale()} Generate Dockerfile for ML fine-tuning.

    Creates a production-ready Dockerfile with CUDA support.
    Optionally generates a FastAPI service template.
    """
    if service and model is None:
        # Generate service Dockerfile and template
        generate_dockerfile(
            model_name=None,
            service=True,
            output_file=output,
            stack=stack,
            base_image=base_image,
            python_version=python_version,
        )
        generate_service_template()
    else:
        generate_dockerfile(
            model_name=model,
            service=service,
            output_file=output,
            stack=stack,
            base_image=base_image,
            python_version=python_version,
        )

    console.print()
    console.print(f"[bold green]{icon_check()} Dockerfile generated![/bold green]")


@app.command(name="test-model")
def test_model_cmd(
    model: str = typer.Argument("tinyllama", help="Model to test (tinyllama, gpt2, mistral-7b)"),
):
    f"""
    {icon_test()} Run smoke test with a real LLM model.

    Tests model loading and forward pass to verify fine-tuning readiness.
    """
    console.print(f"[bold blue]{icon_test()} Testing model: {model}[/bold blue]\n")
    success = gpu_test_model(model_name=model)
    if success:
        console.print()
        console.print(f"[bold green]{icon_check()} Model test passed! Ready for fine-tuning.[/bold green]")
    else:
        console.print()
        console.print(f"[bold red]{icon_cross()} Model test failed. Check diagnostics.[/bold red]")
        raise typer.Exit(1)


@app.command()
def smoke_test():
    f"""
    {icon_test()} Run LoRA fine-tuning smoke test.

    Performs a minimal LoRA fine-tuning test to verify environment setup.
    """
    console.print(f"[bold blue]{icon_test()} Running LoRA smoke test...[/bold blue]\n")
    success = smoke_test_lora()
    if success:
        console.print()
        console.print(f"[bold green]{icon_check()} Smoke test passed! Environment is ready.[/bold green]")
    else:
        console.print()
        console.print(
            f"[bold red]{icon_cross()} Smoke test failed. Run 'mlenvdoctor diagnose' for details.[/bold red]"
        )
        raise typer.Exit(1)


@mcp_app.command("serve")
def mcp_serve():
    """Serve a minimal JSON-line MCP stub."""
    raise typer.Exit(serve_mcp())


app.add_typer(mcp_app, name="mcp")
app.add_typer(stack_app, name="stack")


def main_cli():
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main_cli()
