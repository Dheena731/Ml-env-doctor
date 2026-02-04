"""Constants used throughout ML Environment Doctor."""

from typing import Final

# Version compatibility
MIN_PYTHON_VERSION: Final[tuple[int, int]] = (3, 8)
MIN_PYTORCH_VERSION: Final[str] = "2.4.0"

# CUDA versions
SUPPORTED_CUDA_VERSIONS: Final[list[str]] = ["12.1", "12.4"]
DEFAULT_CUDA_VERSION: Final[str] = "12.4"

# ML Library versions
MIN_TRANSFORMERS_VERSION: Final[str] = "4.44.0"
MIN_PEFT_VERSION: Final[str] = "0.12.0"
MIN_TRL_VERSION: Final[str] = "0.9.0"
MIN_DATASETS_VERSION: Final[str] = "2.20.0"
MIN_ACCELERATE_VERSION: Final[str] = "1.0.0"

# Memory requirements (GB)
MIN_GPU_MEMORY_GB: Final[int] = 8
RECOMMENDED_GPU_MEMORY_GB: Final[int] = 16
MIN_DISK_SPACE_GB: Final[int] = 50

# Timeouts (seconds)
DEFAULT_COMMAND_TIMEOUT: Final[int] = 30
DEFAULT_NETWORK_TIMEOUT: Final[int] = 10
DEFAULT_INSTALL_TIMEOUT: Final[int] = 600

# File paths
DEFAULT_CONFIG_FILE: Final[str] = "mlenvdoctor.toml"
DEFAULT_REQUIREMENTS_FILE: Final[str] = "requirements-mlenvdoctor.txt"
DEFAULT_DOCKERFILE: Final[str] = "Dockerfile.mlenvdoctor"

# Model names
SUPPORTED_MODELS: Final[dict[str, str]] = {
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "gpt2": "gpt2",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
}

# ML Stacks
ML_STACKS: Final[list[str]] = ["trl-peft", "minimal"]

# Diagnostic check names
CHECK_CUDA_DRIVER: Final[str] = "cuda_driver"
CHECK_PYTORCH_CUDA: Final[str] = "pytorch_cuda"
CHECK_ML_LIBRARIES: Final[str] = "ml_libraries"
CHECK_GPU_MEMORY: Final[str] = "gpu_memory"
CHECK_DISK_SPACE: Final[str] = "disk_space"
CHECK_DOCKER_GPU: Final[str] = "docker_gpu"
CHECK_INTERNET: Final[str] = "internet"

# Severity levels
SEVERITY_CRITICAL: Final[str] = "critical"
SEVERITY_WARNING: Final[str] = "warning"
SEVERITY_INFO: Final[str] = "info"

# Status values
STATUS_PASS: Final[str] = "PASS"
STATUS_FAIL: Final[str] = "FAIL"
STATUS_WARN: Final[str] = "WARN"
STATUS_INFO: Final[str] = "INFO"
