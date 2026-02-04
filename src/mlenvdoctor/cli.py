"""CLI entrypoint for ML Environment Doctor."""

from pathlib import Path
from typing import Optional

import typer

from . import __version__
from .diagnose import diagnose_env, print_diagnostic_table
from .dockerize import generate_dockerfile, generate_service_template
from .export import export_csv, export_html, export_json
from .fix import auto_fix
from .gpu import benchmark_gpu_ops, smoke_test_lora, test_model as gpu_test_model
from .icons import icon_check, icon_cross, icon_search, icon_test, icon_whale, icon_wrench
from .logger import get_default_log_file, setup_logger
from .utils import console

app = typer.Typer(
    name="mlenvdoctor",
    help=f"{icon_search()} ML Environment Doctor - Diagnose & fix ML environments for LLM fine-tuning",
    add_completion=False,
)


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
    # Set up logging
    log_path = log_file or get_default_log_file()
    setup_logger(log_file=log_path, level=log_level)


@app.command()
def diagnose(
    full: bool = typer.Option(
        False, "--full", "-f", help="Run full diagnostics including GPU benchmarks"
    ),
    json_output: Optional[Path] = typer.Option(
        None, "--json", help="Export results to JSON file"
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
    issues = diagnose_env(full=full)
    print_diagnostic_table(issues)

    # Export to formats if requested
    if json_output:
        export_json(issues, json_output)
        console.print(f"[green]{icon_check()} Exported to {json_output}[/green]")
    if csv_output:
        export_csv(issues, csv_output)
        console.print(f"[green]{icon_check()} Exported to {csv_output}[/green]")
    if html_output:
        export_html(issues, html_output)
        console.print(f"[green]{icon_check()} Exported to {html_output}[/green]")

    if full:
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
        console.print("[yellow]💡 Run 'mlenvdoctor diagnose' to verify fixes[/yellow]")


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


def main_cli():
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main_cli()
