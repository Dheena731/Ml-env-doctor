"""GPU benchmarks and smoke tests for ML Environment Doctor."""

import time
from typing import Dict

try:
    import torch
except ImportError:
    torch = None  # type: ignore

import sys

from rich.console import Console

from .utils import print_error, print_info, print_success

# Configure console for Windows compatibility
if sys.platform == "win32":
    console = Console(legacy_windows=True, force_terminal=True)
else:
    console = Console()


def benchmark_gpu_ops() -> Dict[str, float]:
    """Run basic GPU operations benchmark."""
    if torch is None or not torch.cuda.is_available():
        return {}

    results = {}
    device = torch.device("cuda:0")

    # Matrix multiplication benchmark
    with console.status("[bold green]Running GPU benchmark..."):
        sizes = [(1024, 1024), (2048, 2048), (4096, 4096)]
        for size in sizes:
            a = torch.randn(size, device=device)
            b = torch.randn(size, device=device)

            # Warmup
            for _ in range(3):
                _ = torch.matmul(a, b)
            torch.cuda.synchronize()

            # Benchmark
            start = time.time()
            for _ in range(10):
                _ = torch.matmul(a, b)
            torch.cuda.synchronize()
            elapsed = (time.time() - start) / 10
            results[f"matmul_{size[0]}x{size[1]}"] = elapsed * 1000  # ms

    return results


def smoke_test_lora() -> bool:
    """Run a LoRA fine-tuning smoke test on dummy data."""
    if torch is None:
        print_error("PyTorch not available")
        return False

    if not torch.cuda.is_available():
        print_error("CUDA not available")
        return False

    try:
        # Try importing required libraries
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import LoraConfig, get_peft_model
        except ImportError as e:
            print_error(f"Required library not available: {e}")
            return False

        print_info("Running LoRA smoke test with tiny model...")

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Use a very small model for smoke test
        model_name = "gpt2"  # Smallest common model
        try:
            with console.status(f"[bold green]Loading {model_name}..."):
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
                ).to(device)

            # Configure LoRA
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["c_attn"],
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM",
            )

            with console.status("[bold green]Applying LoRA..."):
                model = get_peft_model(model, lora_config)

            # Create dummy input
            dummy_text = "Hello, this is a test"
            inputs = tokenizer(dummy_text, return_tensors="pt").to(device)

            # Forward pass
            with console.status("[bold green]Running forward pass..."):
                with torch.no_grad():
                    _ = model(**inputs)

            print_success("LoRA smoke test passed!")
            return True

        except Exception as e:
            print_error(f"LoRA smoke test failed: {e}")
            return False

    except Exception as e:
        print_error(f"Smoke test error: {e}")
        return False


def test_model(model_name: str = "tinyllama") -> bool:
    """Test a specific model for fine-tuning readiness."""
    model_map = {
        "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "gpt2": "gpt2",
        "mistral-7b": "mistralai/Mistral-7B-v0.1",
    }

    actual_model_name = model_map.get(model_name.lower(), model_name)

    if torch is None:
        print_error("PyTorch not available")
        return False

    if not torch.cuda.is_available():
        print_error("CUDA not available")
        return False

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        console.print(f"[bold blue]Testing model: {actual_model_name}[/bold blue]")

        device = torch.device("cuda:0")
        free_mem, total_mem = torch.cuda.mem_get_info(0)
        free_gb = free_mem / (1024**3)

        # Estimate memory requirements (rough)
        if "7b" in actual_model_name.lower() or "7B" in actual_model_name:
            if free_gb < 16:
                print_error(
                    f"Insufficient GPU memory: {free_gb:.1f}GB free, need ~16GB for 7B model"
                )
                return False

        with console.status(f"[bold green]Loading {actual_model_name}..."):
            try:
                tokenizer = AutoTokenizer.from_pretrained(actual_model_name)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                model = AutoModelForCausalLM.from_pretrained(
                    actual_model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )

                # Test forward pass
                dummy_text = "Test"
                inputs = tokenizer(dummy_text, return_tensors="pt").to(device)

                with torch.no_grad():
                    _ = model(**inputs)

                print_success(f"Model {actual_model_name} loaded and tested successfully!")
                return True

            except Exception as e:
                print_error(f"Failed to load/test model: {e}")
                return False

    except ImportError:
        print_error("transformers library not available. Run: pip install transformers")
        return False
    except Exception as e:
        print_error(f"Model test error: {e}")
        return False
