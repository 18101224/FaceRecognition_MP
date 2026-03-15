FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel

# Default arch list is intentionally conservative so rpe_ops build is portable.
# If your torch/cpp_extension supports sm_120, you can override at build:
#   --build-arg TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0+PTX"
ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0+PTX"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONUNBUFFERED=1 \
    FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    build-essential \
    ninja-build \
    cmake \
    pkg-config \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install -U pip setuptools wheel

COPY requirements.txt /workspace/requirements.txt

# Freeze torch stack from base image so requirements cannot override it.
RUN python - <<'PY'
import importlib.metadata as md
from pathlib import Path
pins = []
for name in ("torch", "torchvision", "torchaudio"):
    try:
        pins.append(f"{name}=={md.version(name)}")
    except md.PackageNotFoundError:
        pass
Path("/tmp/torch.constraints.txt").write_text("\n".join(pins) + "\n")
print("[torch constraints]")
print(Path("/tmp/torch.constraints.txt").read_text())
PY

# Remove torch-related requirements lines to keep base torch/cuda pairing intact.
RUN python - <<'PY'
from pathlib import Path
src = Path("/workspace/requirements.txt").read_text().splitlines()
filtered = []
for line in src:
    s = line.strip()
    if not s or s.startswith("#"):
        continue
    lo = s.lower()
    if lo.startswith("--extra-index-url") and "pytorch" in lo:
        continue
    if lo.startswith("torch==") or lo.startswith("torch>=") or lo.startswith("torch<=") or lo == "torch":
        continue
    if lo.startswith("torchvision==") or lo.startswith("torchvision>=") or lo.startswith("torchvision<=") or lo == "torchvision":
        continue
    if lo.startswith("torchaudio==") or lo.startswith("torchaudio>=") or lo.startswith("torchaudio<=") or lo == "torchaudio":
        continue
    filtered.append(s)
Path("/tmp/requirements.no_torch.txt").write_text("\n".join(filtered) + "\n")
print("[requirements.no_torch.txt]")
print(Path("/tmp/requirements.no_torch.txt").read_text())
PY

RUN python -m pip install -r /tmp/requirements.no_torch.txt -c /tmp/torch.constraints.txt

COPY . /workspace

# Build KP-RPE CUDA extension.
RUN python -m pip install -v --no-build-isolation ./models/vit_kprpe/RPE/rpe_ops

RUN python - <<'PY'
import torch
import cv2
import rpe_index_cpp
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available during build:", torch.cuda.is_available())
print("cv2:", cv2.__version__)
print("rpe_index_cpp:", rpe_index_cpp.version())
PY
