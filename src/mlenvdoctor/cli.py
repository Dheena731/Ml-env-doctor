"""CLI entrypoint for ML Environment Doctor."""

from typing import Optional

import typer

from . import __version__
from .diagnose import diagnose_env, print_diagnostic_table
from .dockerize import generate_dockerfile, generate_service_template
from .fix import auto_fix
from .gpu import benchmark_gpu_ops, smoke_test_lora, test_model as gpu_test_model
from .utils import console

app = typer.Typer(
    name="mlenvdoctor",
    help="🔍 ML Environment Doctor - Diagnose & fix ML environments for LLM fine-tuning",
    add_completion=False,
)


def version_callback(value: bool):
    """Print version and exit."""
    if value:
        console.print(f"[bold blue]ML Environment Doctor[/bold blue] version [cyan]{__version__}[/cyan]")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", "-v", callback=version_callback, is_eager=True, help="Show version and exit"
    ),
):
    """ML Environment Doctor - Diagnose & fix ML environments for LLM fine-tuning."""
    pass


@app.command()
def diagnose(
    full: bool = typer.Option(False, "--full", "-f", help="Run full diagnostics including GPU benchmarks"),
):
    """
    🔍 Diagnose your ML environment.

    Quick scan: Checks CUDA, PyTorch, and required ML libraries.
    Full scan (--full): Also checks GPU memory, disk space, Docker GPU support, and connectivity.
    """
    issues = diagnose_env(full=full)
    print_diagnostic_table(issues)

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
    """
    🔧 Auto-fix environment issues and generate requirements.

    Generates requirements.txt or conda environment file based on detected issues.
    Optionally creates a virtual environment and installs dependencies.
    """
    success = auto_fix(use_conda=conda, create_venv=venv, stack=stack)
    if success:
        console.print()
        console.print("[bold green]✅ Auto-fix completed![/bold green]")
        console.print("[yellow]💡 Run 'mlenvdoctor diagnose' to verify fixes[/yellow]")


@app.command()
def dockerize(
    model: Optional[str] = typer.Argument(None, help="Model name (mistral-7b, tinyllama, gpt2)"),
    service: bool = typer.Option(False, "--service", "-s", help="Generate FastAPI service template"),
    output: str = typer.Option("Dockerfile.mlenvdoctor", "--output", "-o", help="Output Dockerfile name"),
):
    """
    🐳 Generate Dockerfile for ML fine-tuning.

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
    console.print("[bold green]✅ Dockerfile generated![/bold green]")


@app.command(name="test-model")
def test_model_cmd(
    model: str = typer.Argument("tinyllama", help="Model to test (tinyllama, gpt2, mistral-7b)"),
):
    """
    🧪 Run smoke test with a real LLM model.

    Tests model loading and forward pass to verify fine-tuning readiness.
    """
    console.print(f"[bold blue]🧪 Testing model: {model}[/bold blue]\n")
    success = gpu_test_model(model_name=model)
    if success:
        console.print()
        console.print("[bold green]✅ Model test passed! Ready for fine-tuning.[/bold green]")
    else:
        console.print()
        console.print("[bold red]❌ Model test failed. Check diagnostics.[/bold red]")
        raise typer.Exit(1)


@app.command()
def smoke_test():
    """
    🧪 Run LoRA fine-tuning smoke test.

    Performs a minimal LoRA fine-tuning test to verify environment setup.
    """
    console.print("[bold blue]🧪 Running LoRA smoke test...[/bold blue]\n")
    success = smoke_test_lora()
    if success:
        console.print()
        console.print("[bold green]✅ Smoke test passed! Environment is ready.[/bold green]")
    else:
        console.print()
        console.print("[bold red]❌ Smoke test failed. Run 'mlenvdoctor diagnose' for details.[/bold red]")
        raise typer.Exit(1)


def main_cli():
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main_cli()

