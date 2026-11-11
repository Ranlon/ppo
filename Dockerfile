# ppo runtime image
#
# The ppo project is pinned to Python 3.11, PyTorch 2.6.0+cu124 and Triton
# 3.2.0.  This base already provides those three components, CUDA headers,
# gcc/g++, and the Python 3.11 development headers needed by Inductor/Triton.
#
# The default registry is the local NVIDIA/PyTorch mirror used on the cluster.
# Override BASE_IMAGE when an equivalent PyTorch 2.6/cu124 devel image is
# available from another approved registry.
# docker.v2.aispeech.com/hpc/ai_bigdata-odb200:torch2.6-cu124
ARG BASE_IMAGE=mirrors.aispeech.com/docker/pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=600 \
    PIP_RETRIES=12 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH=/opt/conda/bin:/usr/local/cuda/bin:/usr/local/nvidia/bin:${PATH}

# The devel base already contains Python 3.11, CUDA headers, gcc/g++, make,
# ninja, cmake, libgomp, and the Python 3.11 development headers.  Avoid an
# apt invocation here: the base's NVIDIA apt repository is not on the cluster
# whitelist, and none of these packages need to be changed for this project.

# PyPI access can be overridden at build time.  The base image already
# contains the exact cu124 Torch/Triton stack, so the install step below
# deliberately removes the lock file's PyTorch extra index.
ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
ENV PIP_INDEX_URL=${PIP_INDEX_URL}

WORKDIR /opt/ppo
COPY . /opt/ppo/

# Install the exact build tools and locked runtime, including the vendored
# legacy LLaMA-Factory PPO compatibility fork.  The base image already has
# torch, torchvision, torchaudio, Triton and the CUDA wheel dependencies at the
# versions required by this project.  Omitting those entries avoids a second
# multi-gigabyte download and avoids the slow official CUDA wheel index.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --prefer-binary -r requirements-build.txt

RUN python - <<'PY'
from pathlib import Path
from packaging.requirements import Requirement

skip = {
    "torch", "torchvision", "torchaudio", "triton",
    "nvidia-cublas-cu12", "nvidia-cuda-cupti-cu12", "nvidia-cuda-nvrtc-cu12",
    "nvidia-cuda-runtime-cu12", "nvidia-cudnn-cu12", "nvidia-cufft-cu12",
    "nvidia-curand-cu12", "nvidia-cusolver-cu12", "nvidia-cusparse-cu12",
    "nvidia-cusparselt-cu12", "nvidia-nccl-cu12", "nvidia-nvjitlink-cu12",
    "nvidia-nvtx-cu12",
}
lines = []
for line in Path("requirements.lock.txt").read_text().splitlines():
    if not line.strip() or line.startswith("#") or line.startswith("--extra-index-url"):
        continue
    if Requirement(line).name.lower() not in skip:
        lines.append(line)
Path("/tmp/ppo-requirements.lock.txt").write_text("\n".join(lines) + "\n")
PY

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --prefer-binary --no-deps \
       --index-url "${PIP_INDEX_URL}" -r /tmp/ppo-requirements.lock.txt \
    && python -m pip install --no-deps --no-build-isolation -e ./vendor/llamafactory

# Fail the image build if the environment no longer matches the project's
# hard checks.  The CPU tests do not require a model, dataset, or GPU.
RUN python - <<'PY'
import importlib.metadata as metadata
import platform
import sys
import torch

assert sys.version_info[:2] == (3, 11), sys.version
assert platform.machine() == "x86_64", platform.machine()
expected = {
    "torch": "2.6.0",
    "torchvision": "0.21.0",
    "torchaudio": "2.6.0",
    "triton": "3.2.0",
    "transformers": "4.57.1",
    "peft": "0.17.1",
    "trl": "0.9.6",
    "accelerate": "1.11.0",
    "datasets": "4.0.0",
    "numpy": "1.26.4",
    "llamafactory": "0.9.4+ppo4.portable1",
}
for name, wanted in expected.items():
    actual = metadata.version(name)
    assert actual.split("+")[0] == wanted.split("+")[0], (name, actual, wanted)
    if "+" in wanted:
        assert actual == wanted, (name, actual, wanted)
assert torch.version.cuda == "12.4", torch.version.cuda
print("ppo dependency check passed")
PY

RUN python -m pip check \
    && python -m compileall -q runtime vendor/llamafactory/src \
    && CUDA_VISIBLE_DEVICES="" DISABLE_VERSION_CHECK=1 \
       python -m unittest discover -s tests -v

# No training command or entrypoint is embedded.  The caller supplies the
# desired docker run command and mounts model/data/output directories.
