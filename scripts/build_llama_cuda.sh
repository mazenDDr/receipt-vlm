#!/usr/bin/env bash
# Build llama.cpp with CUDA for the Blackwell GPU (sm_120), next to the existing CPU build.
# The Windows driver supports CUDA <= 13.1, so the compiler is CUDA 12.9 from NVIDIA's conda channel, installed into
# the "quant" env without changing its existing packages (no sudo needed).
# Run from the project root:  GPU_SESSION=claude-build ./gpu run bash scripts/build_llama_cuda.sh
set -euo pipefail

CONDA=~/miniconda3/bin/conda
QUANT=~/miniconda3/envs/quant
LLAMA=~/llama.cpp

if [ ! -x "$QUANT/bin/nvcc" ]; then
  "$CONDA" install -n quant -y --freeze-installed -c nvidia/label/cuda-12.9.1 cuda-toolkit
fi
"$QUANT/bin/nvcc" --version | tail -2
~/miniconda3/envs/quant/bin/python -c "import torch, transformers; print('quant still:', torch.__version__, transformers.__version__)"

PATH="$QUANT/bin:$PATH" cmake -S "$LLAMA" -B "$LLAMA/build-cuda" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCUDAToolkit_ROOT="$QUANT" -DCMAKE_BUILD_TYPE=Release
PATH="$QUANT/bin:$PATH" cmake --build "$LLAMA/build-cuda" -j 20 \
  --target llama-mtmd-cli llama-quantize llama-server llama-bench
ls "$LLAMA/build-cuda/bin"
