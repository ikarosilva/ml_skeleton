# Dockerfile for explr training environment
# Supports NVIDIA RTX 5090 (Blackwell) with CUDA 12.8 / Driver 570.x

FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip

# Set working directory
WORKDIR /workspace

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the framework
COPY . .

# Install the framework in development mode
RUN pip install -e .

# Create directories for MLflow
RUN mkdir -p /workspace/mlruns /workspace/checkpoints /workspace/artifacts

# Expose ports
# 5000: MLflow UI
# 8888: Jupyter (optional)
EXPOSE 5000 8888

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV MLFLOW_TRACKING_URI=http://localhost:5000
ENV TF_ENABLE_ONEDNN_OPTS=0

# Default command
CMD ["bash"]
