# Multi-stage Dockerfile for LLM inference server with CUDA support

# Stage 1: Builder - Build llama-cpp-python with CUDA support
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_DOCKER_ARCH=all
ENV LLAMA_CUBLAS=1

# Install build dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3-pip \
    build-essential \
    cmake \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools
RUN python3.12 -m pip install --upgrade pip setuptools wheel

# Build and install llama-cpp-python with CUDA support
# This is the most time-consuming step, so we do it separately for better caching
RUN CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --no-cache-dir

# Stage 2: Runtime - Smaller image with only runtime dependencies
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python runtime
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Copy built llama-cpp-python from builder
COPY --from=builder /usr/local/lib/python3.12/dist-packages /usr/local/lib/python3.12/dist-packages

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ /app/app/
COPY scripts/ /app/scripts/

# Create directories for models and logs
RUN mkdir -p /app/models /app/logs

# Expose port
EXPOSE 1133

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:1133/health || exit 1

# Run the server
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "1133"]
