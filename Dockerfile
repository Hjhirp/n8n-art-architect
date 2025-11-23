# Use PyTorch 2.4.0 with CUDA 12.1 and cuDNN 8 as the base
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn8-devel

# 1. System Dependencies & Node.js (Required for n8n-mcp)
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    curl git build-essential \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g npx \
    && rm -rf /var/lib/apt/lists/*

# 2. Install UV for fast Python package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# 3. Set up Workspace
WORKDIR /app
ENV PYTHONPATH=/app/src

# 4. Install Dependencies
# A. Install Unsloth directly via pip for system-wide access
RUN pip install --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# B. Install Python Project Deps via UV
# We define dependencies directly here to ensure Linux/GPU compatibility
# and avoid cross-platform locking issues on macOS
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

# 5. Copy Application Code
COPY src/ ./src/

# 6. Create directories for results
RUN mkdir -p results checkpoints

# 7. Entrypoint
CMD ["python", "src/train.py"]