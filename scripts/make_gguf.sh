#!/usr/bin/env bash
# GGUF versions of the merged fine-tuned model for llama.cpp: a lossless bf16 text model plus the vision projector
# (mmproj, f16), then lower bit widths with llama-quantize, to find where accuracy starts to break.
# Runs in the "quant" env: the converter starts there as is (its pinned requirements would downgrade transformers).
# Usage (project root):
#   GPU_SESSION=claude-gguf ./gpu run bash scripts/make_gguf.sh models/qlora-r16-lm-lr4e4-merged-bf16
set -euo pipefail

MERGED=${1:?merged bf16 model dir}
OUT=${2:-models/$(basename "$MERGED" -merged-bf16)-gguf}
PY=~/miniconda3/envs/quant/bin/python
LLAMA=~/llama.cpp
QUANTIZE=$LLAMA/build-cuda/bin/llama-quantize

mkdir -p "$OUT"
if [ ! -f "$OUT/model-bf16.gguf" ]; then
  CUDA_VISIBLE_DEVICES="" $PY $LLAMA/convert_hf_to_gguf.py "$MERGED" --outtype bf16 --outfile "$OUT/model-bf16.gguf"
fi
# The projector must be f32. Built at f16 it overflows (f16 saturates at 65504 where the bf16 the model was
# trained in does not) on particular receipts: the vision embeddings go NaN and llama.cpp emits token 0 ("!")
# to the generation cap. It is content-dependent and silent — the server logs normal throughput throughout.
# Measured on dev 0000-0009: f16 projector 6/10 receipts flooded, field F1 0.540; f32 projector 0/10, F1 0.899.
# f32 doubles the projector to 2.49 GiB and it cannot be quantized, so it dominates every GGUF row's footprint.
if [ ! -f "$OUT/mmproj-f32.gguf" ]; then
  CUDA_VISIBLE_DEVICES="" $PY $LLAMA/convert_hf_to_gguf.py "$MERGED" --mmproj --outtype f32 --outfile "$OUT/mmproj-f32.gguf"
fi
for q in Q8_0 Q6_K Q5_K_M Q4_K_M Q3_K_M Q2_K; do
  [ -f "$OUT/model-$q.gguf" ] || $QUANTIZE "$OUT/model-bf16.gguf" "$OUT/model-$q.gguf" "$q"
done
ls -la "$OUT"
