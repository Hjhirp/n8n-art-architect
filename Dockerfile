# We use the standard PyTorch 2.4.0 as a lightweight base.
# We will let the installation process handle the upgrade to Nightly (2.7) required by ART.
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

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
# 2. Disable expandable_segments. 
# This is REQUIRED because ART will force-install PyTorch 2.7 (Nightly),
# which crashes vLLM without this flag.
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

# 4. Install Dependencies
# We install Unsloth first. Using [colab-new] pulls the Nightly version,
# which pre-aligns us with the version ART will eventually demand.
RUN pip install --upgrade pip && \
    pip install --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# 5. Install Project Dependencies via UV
# We define dependencies directly here to ensure Linux/GPU compatibility
# This will install vLLM 0.9.2 and confirm PyTorch 2.7 is present.
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