#!/usr/bin/env bash
# Score every GGUF bit width on one split, one model at a time (16 GB of VRAM, one server at a time).
#
#   scripts/score_gguf.sh dev
#
# Each bit width gets its own run dir, so a bit width that fails can be rerun on its own with
# scripts/run_infer.py --resume.
set -euo pipefail

SPLIT=${1:-dev}
GGUF_DIR=${GGUF_DIR:-models/qlora-r16-lm-lr4e4-gguf}
# f32, never f16: an f16 projector floods "!" tokens on some receipts (see scripts/make_gguf.sh).
MMPROJ="$GGUF_DIR/mmproj-f32.gguf"
PYTHON=${PYTHON:-python}
# Widest first: if VRAM or time runs out, the rows already scored are the ones nearest the reference.
WIDTHS=${WIDTHS:-"bf16 Q8_0 Q6_K Q5_K_M Q4_K_M Q3_K_M Q2_K"}

[ -f "$MMPROJ" ] || { echo "missing vision projector: $MMPROJ" >&2; exit 1; }

for width in $WIDTHS; do
  model="$GGUF_DIR/model-$width.gguf"
  if [ ! -f "$model" ]; then
    echo "skip $width: $model not built" >&2
    continue
  fi
  variant="ft-r16-gguf-$(echo "$width" | tr '[:upper:]' '[:lower:]')"
  echo "=== $variant on $SPLIT ($(du -h "$model" | cut -f1)) ==="
  "$PYTHON" scripts/run_infer.py \
    --backend llamacpp \
    --model "$model" \
    --mmproj "$MMPROJ" \
    --variant "$variant" \
    --split "$SPLIT" \
    --tag final
done
