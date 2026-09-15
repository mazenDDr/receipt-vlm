#!/usr/bin/env bash
# One-time setup on the GPU machine for quantization (AWQ + GGUF), kept out of the training env:
#   - a conda env "quant" cloned from "main" (same torch build) holds llm-compressor, which upgrades transformers,
#     so "main" stays at the version every training run used;
#   - llama.cpp is built for the CPU (GGUF is the CPU/edge format), with cmake from pip (no sudo needed).
# Run from the project root:  GPU_SESSION=claude-setup ./gpu run bash scripts/setup_quant_env.sh
set -euo pipefail

CONDA=~/miniconda3/bin/conda
QUANT=~/miniconda3/envs/quant
LLAMA=~/llama.cpp

if [ ! -d "$QUANT" ]; then
  "$CONDA" create --name quant --clone main -y
fi
"$QUANT/bin/pip" install -q llmcompressor cmake
"$QUANT/bin/python" -c "import torch, transformers, llmcompressor as lc; print('quant:', torch.__version__, transformers.__version__, lc.__version__)"
~/miniconda3/envs/main/bin/python -c "import transformers; print('main still:', transformers.__version__)"

if [ ! -d "$LLAMA" ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA"
fi
echo "llama.cpp commit: $(git -C "$LLAMA" rev-parse --short HEAD)"
PATH="$QUANT/bin:$PATH" cmake -S "$LLAMA" -B "$LLAMA/build" -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
PATH="$QUANT/bin:$PATH" cmake --build "$LLAMA/build" -j 20 --target llama-mtmd-cli llama-quantize llama-server
ls "$LLAMA/build/bin"
