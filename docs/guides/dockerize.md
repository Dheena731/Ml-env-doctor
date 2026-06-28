# Dockerize

Generate a starter Dockerfile for common models/stacks.

```bash
mlenvdoctor dockerize tinyllama
mlenvdoctor dockerize mistral-7b --stack llm-training
mlenvdoctor dockerize gpt2 --python-version 3.10 -o Dockerfile.gpt2
```

Service profile (FastAPI scaffold):

```bash
mlenvdoctor dockerize --service --base-image nvidia/cuda:12.4.0-runtime-ubuntu22.04
```

If GPU containers fail while the host GPU works, run:

```bash
mlenvdoctor diagnose --full
```
