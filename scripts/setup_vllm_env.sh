#!/usr/bin/env bash
# One-time setup on the GPU machine: a fresh conda env "vllm" to run the AWQ (W4A16) checkpoints with vLLM's
# 4-bit kernels. vLLM brings its own torch (2.13 with a CUDA 13.0 runtime, within the Windows driver's CUDA 13.1),
# so it can't share "main" or "quant". transformers can't load these checkpoints (two compressed-tensors reader
# bugs) and would decompress them to bf16 anyway.
# Run from the project root:  GPU_SESSION=claude-vllm-setup ./gpu run bash scripts/setup_vllm_env.sh
set -euo pipefail

CONDA=~/miniconda3/bin/conda
VLLM=~/miniconda3/envs/vllm

if [ ! -d "$VLLM" ]; then
  "$CONDA" create --name vllm python=3.11 -y
fi
# vLLM, plus the small packages this project's runner and scoring import
"$VLLM/bin/pip" install -q "vllm==0.29.0" pydantic pyyaml orjson zss
"$VLLM/bin/python" - <<'EOF'
import torch, transformers, vllm
print("vllm", vllm.__version__, "| torch", torch.__version__, "| cuda", torch.version.cuda, "| transformers", transformers.__version__)
print("gpu", torch.cuda.get_device_name(0), "| capability", torch.cuda.get_device_capability(0))
print("sm_120 in torch arch list:", "sm_120" in torch.cuda.get_arch_list(), torch.cuda.get_arch_list())
EOF
~/miniconda3/envs/main/bin/python -c "import torch, transformers; print('main still:', torch.__version__, transformers.__version__)"
