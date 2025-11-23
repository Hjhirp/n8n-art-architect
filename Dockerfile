# GPU-ready base: PyTorch + CUDA 12.1 + cuDNN9
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive

# 1. System deps & Node.js (if you still need Node)
RUN apt-get update && apt-get install -y \
    curl git build-essential \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# 2. Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# 3. Workspace
WORKDIR /app
ENV PYTHONPATH=/app/src

# 4. Critical runtime settings
# Use GNU threading for MKL (recommended for vLLM / PyTorch combos)
ENV MKL_THREADING_LAYER=GNU
# IMPORTANT: do NOT set PYTORCH_CUDA_ALLOC_CONF here.
# Leaving it at default avoids the "torch.cuda.MemPool doesn't currently support expandable_segments" crash.

# 5. Base pip upgrade
RUN pip install --upgrade pip

# 6. Install ART / Unsloth / vLLM environment directly via UV
# This is your provided env, made explicit.
RUN uv pip install --system \
    "openpipe-art[backend]==0.4.11" \
    "peft>=0.14.0" \
    "hf-xet>=1.1.0" \
    "bitsandbytes>=0.45.2" \
    "unsloth==2025.10.3" \
    "unsloth-zoo==2025.10.3" \
    "vllm>=0.9.2,<=0.10.0" \
    "torchtune" \
    "trl>=0.19.0" \
    "torch>=2.7.0" \
    "torchao>=0.9.0" \
    "accelerate==1.7.0" \
    "awscli>=1.38.1" \
    "setproctitle>=1.3.6" \
    "tblib>=3.0.0" \
    "setuptools>=78.1.0" \
    "wandb==0.21.0" \
    "polars>=1.26.0" \
    "transformers==4.53.2" \
    "trl==0.20.0" \
    "nbclient>=0.10.1" \
    "pytest>=8.4.1" \
    "nbmake>=1.5.5" \
    "gql<4" \
    "mcp>=1.11.0" \
    "openai" \
    "python-dotenv" \
    "pydantic" \
    "tenacity"

# 7. Copy your code
COPY src/ ./src/

# 8. Result/checkpoint dirs
RUN mkdir -p results checkpoints

# 9. Entrypoint
CMD ["python", "src/train.py"]
