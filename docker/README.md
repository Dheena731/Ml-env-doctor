# Docker Support with ML Environment Doctor

ML Environment Doctor helps you create production-ready Docker images for ML fine-tuning with proper CUDA support, dependency management, and environment validation.

## 🚀 Quick Start

### Generate and Build Your First Docker Image

```bash
# 1. Diagnose your local environment first
mlenvdoctor diagnose

# 2. Generate a Dockerfile for your model
mlenvdoctor dockerize mistral-7b

# 3. Build the Docker image
docker build -f Dockerfile.mlenvdoctor -t my-ml-model .

# 4. Run with GPU support
docker run --gpus all my-ml-model
```

## 📋 Complete Workflow

### Step 1: Diagnose Your Environment

Before generating Dockerfiles, verify your local setup:

```bash
# Quick diagnostic scan
mlenvdoctor diagnose

# Full scan with GPU benchmarks and extended checks
mlenvdoctor diagnose --full

# Export diagnostics for documentation
mlenvdoctor diagnose --full --json diagnostics.json --html report.html
```

**What this checks:**
- ✅ NVIDIA GPU driver and CUDA version
- ✅ PyTorch CUDA compatibility
- ✅ Required ML libraries (transformers, peft, trl, etc.)
- ✅ GPU memory availability
- ✅ Disk space for model cache
- ✅ Docker GPU support (nvidia-docker)
- ✅ Internet connectivity for Hugging Face Hub

### Step 2: Generate Requirements (Optional but Recommended)

Ensure consistent dependencies between local and container environments:

```bash
# Generate requirements.txt with detected stack
mlenvdoctor fix

# Or specify a stack
mlenvdoctor fix --stack trl-peft    # Full stack for fine-tuning
mlenvdoctor fix --stack minimal     # Minimal dependencies

# Generate conda environment file
mlenvdoctor fix --conda
```

### Step 3: Generate Dockerfile

Create production-ready Dockerfiles optimized for your use case:

```bash
# For model fine-tuning
mlenvdoctor dockerize mistral-7b
mlenvdoctor dockerize tinyllama
mlenvdoctor dockerize gpt2

# For FastAPI service deployment
mlenvdoctor dockerize --service

# Custom output location
mlenvdoctor dockerize mistral-7b -o docker/Dockerfile.mistral
```

**Generated Dockerfiles include:**
- ✅ NVIDIA CUDA 12.4 base image
- ✅ PyTorch with CUDA support
- ✅ All required ML libraries
- ✅ Optimized layer caching
- ✅ Proper GPU runtime configuration
- ✅ Health checks (for services)

### Step 4: Build and Run

```bash
# Build the image
docker build -f Dockerfile.mlenvdoctor -t mlenvdoctor:latest .

# Run with GPU support
docker run --gpus all mlenvdoctor:latest

# Run with GPU and mount data directory
docker run --gpus all -v $(pwd)/data:/app/data mlenvdoctor:latest

# Run FastAPI service
docker run --gpus all -p 8000:8000 mlenvdoctor:latest
```

## 🎯 Use Cases

### Use Case 1: Model Fine-Tuning Container

```bash
# Complete workflow
mlenvdoctor diagnose --full --html pre-build-report.html
mlenvdoctor dockerize mistral-7b -o Dockerfile.training
docker build -f Dockerfile.training -t mistral-training .
docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output mistral-training
```

### Use Case 2: FastAPI Service Deployment

```bash
# Generate service template
mlenvdoctor dockerize --service

# This creates:
# - Dockerfile.mlenvdoctor (with FastAPI/uvicorn)
# - app.py (FastAPI service template)

# Build and run
docker build -f Dockerfile.mlenvdoctor -t ml-api .
docker run --gpus all -p 8000:8000 ml-api
```

### Use Case 3: CI/CD Integration

```bash
# In your CI pipeline
mlenvdoctor diagnose --json ci-diagnostics.json
mlenvdoctor dockerize tinyllama -o Dockerfile.ci
docker build -f Dockerfile.ci -t test-image .
docker run --gpus all test-image python -m pytest
```

## 🔧 Advanced Features

### Export Diagnostics

Archive diagnostics alongside your Dockerfiles for documentation and debugging:

```bash
# Export to multiple formats
mlenvdoctor diagnose --full \
  --json docker-diagnostics.json \
  --csv docker-diagnostics.csv \
  --html docker-report.html

# Use in CI/CD
mlenvdoctor diagnose --json diagnostics-$(date +%Y%m%d).json
```

### Logging

Enable detailed logging for troubleshooting Docker builds:

```bash
# With custom log file
mlenvdoctor diagnose --log-file docker-build.log --log-level DEBUG

# Logs include:
# - Diagnostic check results
# - Error details
# - Performance metrics
# - GPU benchmark results
```

### Environment Validation

Test your environment before building Docker images:

```bash
# Quick validation
mlenvdoctor diagnose

# Full validation with GPU test
mlenvdoctor diagnose --full
mlenvdoctor test-model tinyllama  # Test actual model loading

# LoRA smoke test
mlenvdoctor smoke-test
```

## 📦 Generated Files

When you run `mlenvdoctor dockerize`, you get:

### For Model Training:
- `Dockerfile.mlenvdoctor` - Production-ready Dockerfile
- Optimized layer caching for faster rebuilds
- CUDA 12.4 support
- All ML dependencies pre-installed

### For FastAPI Service:
- `Dockerfile.mlenvdoctor` - Service-ready Dockerfile
- `app.py` - FastAPI service template
- Health check endpoint
- Port 8000 exposed

## 🐳 Docker Best Practices

### 1. Multi-Stage Builds (Manual Enhancement)

The generated Dockerfile is a solid base. For production, consider multi-stage builds:

```dockerfile
# Build stage
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS builder
# ... install dependencies ...

# Runtime stage
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04
COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
# ... copy only runtime files ...
```

### 2. Layer Caching

The generated Dockerfiles are optimized for layer caching. Dependencies are installed before your code:

```dockerfile
# Dependencies (cached layer)
RUN pip install torch transformers peft ...

# Your code (changes frequently)
COPY . /app
```

### 3. GPU Memory Management

For large models, adjust memory settings:

```bash
docker run --gpus all \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
  my-ml-model
```

### 4. Volume Mounts

Mount data and outputs:

```bash
docker run --gpus all \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/models:/app/models \
  my-ml-model
```

## 🔍 Troubleshooting

### Issue: "CUDA not available in container"

**Solution:**
```bash
# Verify nvidia-docker is installed
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

# Check Docker GPU support
mlenvdoctor diagnose --full  # Should show "Docker GPU Support: PASS"
```

### Issue: "Out of memory errors"

**Solution:**
```bash
# Check GPU memory before building
mlenvdoctor diagnose --full  # Shows available GPU memory

# Use smaller batch sizes or model quantization
# Consider using bitsandbytes for 8-bit/4-bit models
```

### Issue: "Slow builds"

**Solution:**
```bash
# Use BuildKit for faster builds
DOCKER_BUILDKIT=1 docker build -f Dockerfile.mlenvdoctor -t my-model .

# Leverage layer caching (dependencies install before code copy)
# The generated Dockerfile already optimizes for this
```

### Issue: "Model download fails in container"

**Solution:**
```bash
# Pre-download models locally and mount cache
docker run --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  my-ml-model

# Or set HF_HOME environment variable
docker run --gpus all -e HF_HOME=/app/models my-ml-model
```

## 📚 Additional Resources

- **Main README**: See `README.md` for complete CLI documentation
- **Contributing**: See `CONTRIBUTING.md` for development guidelines
- **PyPI Package**: https://pypi.org/project/mlenvdoctor/
- **Docker GPU Guide**: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/

## 💡 Tips

1. **Always diagnose first**: Run `mlenvdoctor diagnose` before generating Dockerfiles to catch issues early
2. **Export diagnostics**: Use `--json` or `--html` to document your environment setup
3. **Test locally**: Use `mlenvdoctor test-model` to verify GPU setup before containerizing
4. **Version pinning**: The generated requirements use `>=` constraints. For production, consider pinning exact versions
5. **Service health**: FastAPI templates include `/health` endpoint for monitoring

---

**Need help?** Open an issue on GitHub or check the main documentation.
