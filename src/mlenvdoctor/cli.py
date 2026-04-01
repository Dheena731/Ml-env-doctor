"""CLI entrypoint for ML Environment Doctor."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
import typer
from typer.core import TyperArgument

from . import __version__
from .diagnose import diagnose_env, get_fix_commands, print_diagnostic_table
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
    {icon_search()} Diagnose your ML environment.

    Quick scan: Checks CUDA, PyTorch, and required ML libraries.
    Full scan (--full): Also checks GPU memory, disk space, Docker GPU support, and connectivity.
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
    {icon_search()} Run diagnostics with optional CI-friendly output.
    """
    issues = diagnose_env(full=full, show_header=not ci)
    exit_code = get_exit_code(issues)

    if ci:
        payload = build_export_data(issues, include_metadata=True)
        summary = payload["summary"]
        typer.echo(
            f"mlenvdoctor status={exit_code} passed={summary['passed']} "
            f"warnings={summary['warnings']} critical={summary['critical']}"
        )
        for fix in payload["fixes"]:
            typer.echo(f"FIX {fix['issue']}: {fix['command']}")
    else:
        print_diagnostic_table(issues)

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
    stack: str = typer.Option("trl-peft", "--stack", "-s", help="ML stack: trl-peft or minimal"),
):
    f"""
    {icon_wrench()} Auto-fix environment issues and generate requirements.

    Generates requirements.txt or conda environment file based on detected issues.
    Optionally creates a virtual environment and installs dependencies.
    """
    success = auto_fix(use_conda=conda, create_venv=venv, stack=stack)
    if success:
        console.print()
        console.print(f"[bold green]{icon_check()} Auto-fix completed![/bold green]")
        console.print("[yellow]Run 'mlenvdoctor diagnose' to verify fixes[/yellow]")


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
        generate_dockerfile(model_name=None, service=True, output_file=output)
        generate_service_template()
    else:
        generate_dockerfile(model_name=model, service=service, output_file=output)

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
