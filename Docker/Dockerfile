FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 AS base
ENV DEBIAN_FRONTEND=noninteractive

# 1) Install OS packages (build tools, python3, python3-venv, python3-pip, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      git \
      curl \
      ca-certificates \
      python3 \
      python3-dev \
      python3-venv \
      python3-pip \
      vim \
      pkg-config \
      libopencv-dev \
      gdb \
      cuda-gdb-12-8 \
      cuda-nvtx-12-8 \
    && rm -rf /var/lib/apt/lists/*

# 2) Create a venv at /opt/venv and upgrade pip inside it
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip

# 3) Copy in requirements.txt (which includes the PyTorch+cu128 lines)
COPY requirements.txt /workspace/requirements.txt

# 4) Use the venv’s pip to install all Python dependencies
RUN /opt/venv/bin/pip install --no-cache-dir -r /workspace/requirements.txt

# 5) Put the venv’s bin directory first in PATH
ENV PATH="/opt/venv/bin:${PATH}"

# 6) Set working directory and switch to non-root
WORKDIR /workspace
USER ubuntu

# 7) Expose port 8888 if you plan to run Jupyter
EXPOSE 8888

# 8) Default command: drop into bash
CMD ["/bin/bash"]
