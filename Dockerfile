# CUDA + Python base. We'll let openpipe-art[backend] pull in torch>=2.7.0, vLLM, unsloth, etc.
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# 1. System deps & Python
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    curl git build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. (Optional) Node.js, if you still need it
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get update && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# 3. Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# 4. Workspace
WORKDIR /app
ENV PYTHONPATH=/app/src

# 5. Critical runtime settings
# Use GNU threading for MKL to avoid weirdness with vLLM + PyTorch
ENV MKL_THREADING_LAYER=GNU
# IMPORTANT: do NOT set PYTORCH_CUDA_ALLOC_CONF here.
# Leaving it at default avoids "torch.cuda.MemPool doesn't currently support expandable_segments".

# 6. Upgrade pip
RUN python3 -m pip install --upgrade pip

# 7. Install openpipe-art with backend extras (this pulls the env in your pyproject.toml)
# - openpipe-art core: openai, typer, litellm, weave
# - backend extra: peft, unsloth, unsloth-zoo, vllm 0.9–0.10, torch>=2.7.0, torchao, accelerate, wandb, etc.
RUN uv pip install --system \
    "openpipe-art[backend]==0.5.2" \
    "openpipe-art[plotting]==0.5.2" \
    "openpipe-art[langgraph]==0.5.2" \
    "openpipe-art[skypilot]==0.5.2" \
    "mcp>=1.11.0" \
    "python-dotenv" \
    "tenacity" \
    "pydantic"

# 8. Copy your training code
COPY src/ ./src/

# 9. Create dirs for results / checkpoints
RUN mkdir -p results checkpoints

# 10. Entrypoint
CMD ["python3", "src/train.py"]
