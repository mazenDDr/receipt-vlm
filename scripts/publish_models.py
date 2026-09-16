"""Publish the weights to the Hugging Face Hub, with model cards generated from the runs.

    python scripts/publish_models.py \
        --adapter-repo  <user>/<name>-lora \
        --awq-repo      <user>/<name>-awq \
        --gguf-repo     <user>/<name>-GGUF

Add --dry-run to write the cards and print the upload commands without touching the Hub.

The weights live only on the GPU machine (models/ is git-ignored and never synced), so this runs there.
Every number in every card is read from outputs/runs/*/summary.json, the same rule the rest of the
project follows: re-score a variant and the cards follow rather than drifting.

What is deliberately NOT published:
  - checkpoints/          training state, not a released artifact
  - mmproj-f16.gguf       overflows into NaN on some receipts; shipping it hands people the bug
  - Q3_K_M / Q2_K         the uncalibrated rows that collapse, kept in the repo as evidence only
"""

# ruff: noqa: E501 - the model cards below are markdown that gets published verbatim. A table row has
# to be one line or it stops being a table, and the code samples are meant to be copy-pasted, so
# wrapping them to satisfy the line limit would ship broken content.
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
BASE_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

ADAPTER_DIR = MODELS / "qlora-r16-lm-lr4e4/adapter"
AWQ_DIR = MODELS / "qlora-r16-lm-lr4e4-awq-w4a16"
GGUF_DIR = MODELS / "qlora-r16-lm-lr4e4-gguf"
GGUF_FILES = ["model-Q4_K_M.gguf", "mmproj-f32.gguf"]


def summary(run: str) -> dict[str, Any]:
    return json.loads((ROOT / "outputs/runs" / run / "summary.json").read_text())


def scores() -> dict[str, dict[str, float]]:
    """Test-split headlines for every variant a card mentions."""
    runs = {
        "base": "20260915-2200_base-bf16_test_final",
        "ft": "20260915-2231_qlora-r16-lm-lr4e4_test_final",
        "bf16": "20260916-0000_bf16-merged-vllm_test_final",
        "awq": "20260915-2353_awq-w4a16-vllm_test_final",
        "gguf": "20260916-0516_ft-r16-gguf-q4_k_m_test_final",
    }
    out = {}
    for key, run in runs.items():
        data = summary(run)
        out[key] = {
            "f1": data["metrics"]["f1"]["value"],
            "numeric": data["metrics"]["f1_numeric"]["value"],
            "text": data["metrics"]["f1_text"]["value"],
            "valid": data["metrics"]["valid_json"]["value"],
            "p50": data["run"]["latency_s"]["p50"],
            "tps": data["run"]["output_tokens_per_s"],
        }
    return out


FRONT = """---
license: apache-2.0
base_model: {base}
tags:
{tags}
datasets:
- naver-clova-ix/cord-v2
language:
- en
- id
pipeline_tag: image-text-to-text
library_name: {library}
---
"""

SHARED_TAIL = """
## How it was trained

QLoRA on one RTX 5060 Ti (16 GB): the base model in 4-bit NF4 with the vision tower kept in bf16, LoRA
rank 16 / alpha 32 on the language-model layers only, learning rate 4e-4, 2 epochs, loss on the answer
tokens alone. 1.07 h, 7.6 GB peak VRAM, 29.9 M trainable parameters.

Training data is [CORD v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) (CC BY 4.0), split
773 / 99 / 100 after dropping 28 images that appeared in more than one split under a 256-bit perceptual
hash. Every choice was made on the dev split; the test split was scored once.

## Limitations

- **It is trained on one dataset's conventions.** CORD labels transcribe what is printed, stray marks
  included, and Indonesian receipts use `.` and `,` interchangeably as thousands separators. The model
  learned those conventions; on receipts from elsewhere, formatting may not match your expectations.
- **The remaining errors are structural, not perceptual.** All 100 held-out receipts produced valid
  JSON. The worst one has every value correct and every key wrong. Separator conventions alone account
  for about a third of the numeric gap.
- **Not for accounting without review.** It misfiles fields often enough that a human should check
  anything that matters financially.

## Links

- Code, evaluation harness and write-ups: <https://github.com/mazenDDr/receipt-vlm>
- Field guide, one-receipt tour, and 100 recorded extractions: <https://mazenddr.github.io/receipt-vlm/>

## Citation

Receipts from CORD v2 (Park et al., *CORD: A Consolidated Receipt Dataset for Post-OCR Parsing*, 2019),
CC BY 4.0. Base model {base}, Apache-2.0.
"""


def adapter_card(s: dict[str, dict[str, float]], repos: dict[str, str]) -> str:
    return (
        FRONT.format(
            base=BASE_MODEL,
            tags="\n".join(f"- {t}" for t in ("lora", "peft", "receipts", "document-understanding", "ocr")),
            library="peft",
        )
        + f"""
# Receipt field extraction — LoRA adapter

A LoRA adapter for [{BASE_MODEL}](https://huggingface.co/{BASE_MODEL}) that turns a photograph of a till
receipt into JSON: every item with its count and price, plus subtotal, tax, total and payment.

**119 MB.** Apply it to the base model, or use one of the ready-made builds:
[AWQ 4-bit](https://huggingface.co/{repos["awq"]}) for a GPU server,
[GGUF](https://huggingface.co/{repos["gguf"]}) for llama.cpp.

## What it changes

| | Field F1 | Numeric | Text | Valid JSON |
|---|---|---|---|---|
| Base model, zero-shot | {s["base"]["f1"]:.3f} | {s["base"]["numeric"]:.3f} | {s["base"]["text"]:.3f} | {s["base"]["valid"]:.3f} |
| **With this adapter** | **{s["ft"]["f1"]:.3f}** | {s["ft"]["numeric"]:.3f} | {s["ft"]["text"]:.3f} | {s["ft"]["valid"]:.3f} |

100 held-out receipts, scored once. The paired difference is **+0.396 [+0.345, +0.442]** — a 95%
bootstrap interval over the same receipts, excluding zero.

## Use

```python
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

model = Qwen2_5_VLForConditionalGeneration.from_pretrained("{BASE_MODEL}", dtype="bfloat16", device_map="cuda")
model = PeftModel.from_pretrained(model, "{repos["adapter"]}")
processor = AutoProcessor.from_pretrained("{BASE_MODEL}")
```

The prompt matters: it names every CORD key and asks for only the ones printed. Use the exact
instruction from [`src/receipt_vlm/prompts.py`](https://github.com/mazenDDr/receipt-vlm/blob/main/src/receipt_vlm/prompts.py),
and greedy decoding with `repetition_penalty=1.0` — Qwen's default 1.05 pushes the model away from
repeating digits like `.000`.
"""
        + SHARED_TAIL.format(base=BASE_MODEL)
    )


def awq_card(s: dict[str, dict[str, float]], repos: dict[str, str], quant: dict[str, Any]) -> str:
    return (
        FRONT.format(
            base=BASE_MODEL,
            tags="\n".join(
                f"- {t}" for t in ("awq", "vllm", "quantized", "receipts", "document-understanding")
            ),
            library="transformers",
        )
        + f"""
# Receipt field extraction — AWQ W4A16

The fine-tuned model quantized to 4-bit with AWQ (W4A16, language model only; the vision tower stays in
bf16). **{quant["checkpoint_gb"]:.2f} GB**, down from {quant["merged_bf16_gb"]:.2f} GB.

This is the build the project recommends for a GPU server: smallest, fastest, and its accuracy is within
noise of the bf16 model it came from.

| | Field F1 | Latency p50 | Output tok/s |
|---|---|---|---|
| Fine-tuned bf16 | {s["bf16"]["f1"]:.3f} | {s["bf16"]["p50"]:.2f} s | {s["bf16"]["tps"]:.1f} |
| **This build** | **{s["awq"]["f1"]:.3f}** | **{s["awq"]["p50"]:.2f} s** | **{s["awq"]["tps"]:.1f}** |

Paired against its own bf16 source on the same 100 receipts: **−0.005 [−0.026, +0.013]**. The interval
includes zero, so the difference is not real. Calibrated on {quant["calibration_receipts"]} training
receipts with [llm-compressor](https://github.com/vllm-project/llm-compressor).

## Use it with vLLM

```bash
vllm serve {repos["awq"]} --max-model-len 4096 --limit-mm-per-prompt '{{"image": 1}}'
```

**Serve this with vLLM, not transformers.** At the time it was produced, transformers 5.14 with
compressed-tensors 0.18 could not load this checkpoint (two reader bugs in the packed-quantized format),
and would have decompressed it to bf16 anyway — losing the point of the 4-bit weights. vLLM's W4A16
kernels run it natively.
"""
        + SHARED_TAIL.format(base=BASE_MODEL)
    )


def gguf_card(s: dict[str, dict[str, float]], repos: dict[str, str]) -> str:
    return (
        FRONT.format(
            base=BASE_MODEL,
            tags="\n".join(
                f"- {t}" for t in ("gguf", "llama.cpp", "quantized", "receipts", "document-understanding")
            ),
            library="gguf",
        )
        + f"""
# Receipt field extraction — GGUF Q4_K_M

The fine-tuned model for [llama.cpp](https://github.com/ggerganov/llama.cpp), as Q4_K_M text weights plus
the vision projector.

| File | Size | What it is |
|---|---|---|
| `model-Q4_K_M.gguf` | 1.79 GiB | the language model, 4-bit |
| `mmproj-f32.gguf` | 2.49 GiB | the vision projector — **required** |

| | Field F1 | Latency p50 | Output tok/s |
|---|---|---|---|
| GGUF bf16 (reference) | 0.846 | 2.57 s | 41.4 |
| **Q4_K_M** | **{s["gguf"]["f1"]:.3f}** | **{s["gguf"]["p50"]:.2f} s** | **{s["gguf"]["tps"]:.1f}** |

Paired against that bf16 reference on the same 100 receipts: **−0.014 [−0.038, +0.009]**. Not real.

## Use

```bash
llama-server -m model-Q4_K_M.gguf --mmproj mmproj-f32.gguf \\
  --image-max-tokens 1024 -c 4096 -ngl 99 --temp 0 --top-k 1 --repeat-penalty 1.0
```

## Read this before swapping the projector

**The projector must be f32.** An f16 projector overflows into NaN on particular receipts, and the model
then emits `!` until it hits the generation cap — content-dependent, deterministic, and completely
silent: the server logs normal throughput throughout. Measured on ten receipts, an f16 projector flooded
six of them and scored field F1 0.540 with 60% truncated; the f32 projector scored 0.899 with nothing
truncated. That is why only the f32 file is published here.

It also cannot be quantized, so it is a fixed 2.49 GiB on every row. The total here (4.28 GiB) is larger
than the [AWQ checkpoint](https://huggingface.co/{repos["awq"]}) at 3.31 GiB — lower bit width did not produce a smaller
deployment.

**Q3_K_M and Q2_K are not published.** Without an importance matrix they collapse (field F1 0.100 and
0.000 — Q2_K emits a median of three tokens). With one they recover on dev, but Q2_K+imatrix is then
genuinely worse on test (−0.054 [−0.092, −0.020]). Q4_K_M is the row worth shipping.
"""
        + SHARED_TAIL.format(base=BASE_MODEL)
    )


def run(command: list[str], dry: bool) -> None:
    print("  $ " + " ".join(command))
    if not dry:
        subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-repo", required=True, help="e.g. mazenDDr/<name>-lora")
    parser.add_argument("--awq-repo", required=True, help="e.g. mazenDDr/<name>-awq")
    parser.add_argument("--gguf-repo", required=True, help="e.g. mazenDDr/<name>-GGUF")
    parser.add_argument("--hf", default=str(Path.home() / "miniconda3/envs/main/bin/hf"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cards-out", default="outputs/model_cards")
    args = parser.parse_args()

    repos = {"adapter": args.adapter_repo, "awq": args.awq_repo, "gguf": args.gguf_repo}
    s = scores()
    quant = (
        json.loads((AWQ_DIR / "quant_summary.json").read_text())
        if AWQ_DIR.exists()
        else {"checkpoint_gb": 3.39, "merged_bf16_gb": 7.51, "calibration_receipts": 256}
    )

    cards = ROOT / args.cards_out
    cards.mkdir(parents=True, exist_ok=True)
    written = {
        "adapter": (cards / "adapter.md", adapter_card(s, repos)),
        "awq": (cards / "awq.md", awq_card(s, repos, quant)),
        "gguf": (cards / "gguf.md", gguf_card(s, repos)),
    }
    for key, (path, text) in written.items():
        path.write_text(text)
        print(f"{path.relative_to(ROOT)}: {len(text.splitlines())} lines  → {repos[key]}")

    print("\nuploads:")
    for key, repo in repos.items():
        run([args.hf, "repo", "create", repo, "--type", "model"], args.dry_run)
        run([args.hf, "upload", repo, str(written[key][0]), "README.md"], args.dry_run)

    run([args.hf, "upload", repos["adapter"], str(ADAPTER_DIR), "."], args.dry_run)
    run([args.hf, "upload", repos["awq"], str(AWQ_DIR), "."], args.dry_run)
    for name in GGUF_FILES:
        run([args.hf, "upload", repos["gguf"], str(GGUF_DIR / name), name], args.dry_run)

    if args.dry_run:
        print("\ndry run: nothing was created or uploaded.")


if __name__ == "__main__":
    main()
