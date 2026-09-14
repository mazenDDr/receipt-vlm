# receipt-vlm

Fine-tuning and quantizing a small vision-language model to read receipts.

**Status: work in progress.** The results will appear here as they are measured.

## The question

General-purpose VLMs can read text, but pulling structured fields out of a crumpled receipt photo is harder:
valid JSON every time, the right price in the right slot, every digit exact. This project measures:

1. How much QLoRA fine-tuning improves **Qwen2.5-VL-3B-Instruct** on receipt field extraction, with confidence intervals.
2. What to adapt: rank, learning rate, and language-model layers only vs. also the vision projector.
3. Where quantization (AWQ 4-bit, GGUF at several bit widths) starts to hurt the task, and whether digits break
   before text.

## Data

[CORD v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) (Park et al., 2019): 1,000 receipt photos with
labeled fields, 800 train / 100 validation / 100 test. Licensed CC BY 4.0.

## Running

Code is written locally and runs on a remote GPU (RTX 5060 Ti, 16 GB) through the `./gpu` helper
(`./gpu push`, `./gpu run`, `./gpu status`, `./gpu pull`), which wraps `rsync`, `ssh` and `tmux`.

```bash
make setup   # local venv with the light dependencies
make check   # lint + unit tests (CPU only)
```
