# Use Torch 2.5.1 with CUDA 12.1 and cuDNN 9, GPU-enabled
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

# 1. System Dependencies & Node.js
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    curl git build-essential \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# 2. Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# 3. Set up Workspace
WORKDIR /app
ENV PYTHONPATH=/app/src

# --- CRITICAL FIXES ---
# 1. Force MKL to use GNU threading to prevent vLLM/PyTorch conflicts
ENV MKL_THREADING_LAYER=GNU
# 2. Do NOT set PYTORCH_CUDA_ALLOC_CONF here.
#    Leaving the default allocator avoids the MemPool + expandable_segments crash.

# 4. Install Dependencies
# Torch 2.5.1 + CUDA 12.1 is ALREADY installed in this base image (GPU-enabled),
# so we DO NOT re-install torch via pip (to avoid CPU-only wheels).
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# 5. Install Project Dependencies via UV
# We don't touch torch here; we just install ART + friends.
RUN uv pip install --system \
    "openpipe-art[backend]==0.4.11" \
    "mcp>=1.11.0" \
    "openai" \
    "python-dotenv" \
    "wandb" \
    "pydantic" \
    "tenacity" \
    "vllm==0.9.2" \
    "triton==3.3.0"

# 6. Copy Application Code
COPY src/ ./src/

# 7. Create directories for results
RUN mkdir -p results checkpoints

# 8. Entrypoint
CMD ["python", "src/train.py"]
