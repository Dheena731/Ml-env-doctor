# Docker Support with ML Environment Doctor

ML Environment Doctor can generate a configurable Dockerfile for training or service workflows. The output is still template-based, but it now supports more explicit inputs so the generated image better matches the target workload.

## Quick Start

```bash
mlenvdoctor diagnose --full
mlenvdoctor dockerize tinyllama
docker build -f Dockerfile.mlenvdoctor -t my-ml-model .
docker run --gpus all my-ml-model
```

## Training Images

```bash
mlenvdoctor dockerize mistral-7b --stack llm-training
mlenvdoctor dockerize gpt2 --stack minimal --python-version 3.10
mlenvdoctor dockerize tinyllama --base-image nvidia/cuda:12.4.0-runtime-ubuntu22.04
```

Inputs you can control:

- model profile
- dependency stack
- CUDA base image
- Python version package
- output filename

## Service Images

```bash
mlenvdoctor dockerize --service
mlenvdoctor dockerize --service --base-image nvidia/cuda:12.4.0-runtime-ubuntu22.04
```

Service mode generates:

- `Dockerfile.mlenvdoctor`
- `app.py` FastAPI starter template

## Recommended Workflow

1. Run `mlenvdoctor diagnose --full`
2. Export a report if you need to share the host machine state
3. Generate the Dockerfile with the stack and base image you actually want
4. Review the generated file before building
5. Build and run with GPU access enabled

## Current Limitations

- the Docker output is still generated from internal templates
- only a small set of model profiles are built in
- no Compose or multi-stage generation is included yet

Use the generated Dockerfile as a solid starting point rather than assuming it fully matches every production requirement.
