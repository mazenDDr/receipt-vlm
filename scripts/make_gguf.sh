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

# K-quants below Q4 need an importance matrix. Without one they do not degrade, they collapse: measured on dev,
# Q3_K_M scored 0.100 with 17% valid JSON and Q2_K emitted a median of 3 tokens. With a matrix built from the
# training receipts, Q3_K_M scores 0.858 and Q2_K 0.876 - the latter statistically tied with bf16 on field F1.
# Both sets are kept: "-imat" versus plain is the evidence for that finding.
IMATRIX=$OUT/imatrix.dat
CALIB=${CALIB:-data/processed/calib.txt}
if [ ! -f "$IMATRIX" ]; then
  [ -f "$CALIB" ] || PYTHONPATH=src $PY scripts/make_calib_text.py --split train --out "$CALIB"
  $LLAMA/build-cuda/bin/llama-imatrix -m "$OUT/model-bf16.gguf" -f "$CALIB" -o "$IMATRIX" -ngl 99 --chunks 200
fi
for q in Q3_K_M Q2_K; do
  [ -f "$OUT/model-$q-imat.gguf" ] ||
    $QUANTIZE --imatrix "$IMATRIX" "$OUT/model-bf16.gguf" "$OUT/model-$q-imat.gguf" "$q"
done
ls -la "$OUT"
