"""Merge a LoRA adapter into the bf16 base, then quantize the language model to 4 bits with AWQ.

Runs in the conda env "quant" (llm-compressor needs a newer transformers than training), from the project
root:
    /home/mazen/miniconda3/envs/quant/bin/python scripts/quantize_awq.py \
        --adapter models/qlora-r16-lm/adapter --out models/qlora-r16-lm-awq-w4a16 [--n-calib 256]

Calibration uses train receipts only (prompt + image + gold answer). Writes <out>/ (compressed checkpoint +
processor + quant_summary.json); the merged bf16 model is kept next to it as <out>-merged-bf16/.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from receipt_vlm.data.build import load_examples
from receipt_vlm.quant import awq
from receipt_vlm.train.collate import Collator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--merged", type=Path, help="merged bf16 model dir (default: models/<run>-merged-bf16)"
    )
    parser.add_argument("--n-calib", type=int, default=256)
    parser.add_argument("--max-seq", type=int, default=2048)
    parser.add_argument("--scheme", default="W4A16_ASYM")
    parser.add_argument("--min-pixels", type=int, default=3136)
    parser.add_argument("--max-pixels", type=int, default=802816)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; models are write-once")

    import torch
    from datasets import Dataset
    from llmcompressor import oneshot
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    t0 = time.perf_counter()
    # one merged bf16 model per adapter, shared by every quantization of it (and a trade-off row itself)
    merged = args.merged or Path("models") / f"{Path(args.adapter).parent.name}-merged-bf16"
    if not merged.exists():
        awq.merge_adapter(args.base, args.adapter, merged)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(merged, dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(merged)
    n_layers = len(model.model.language_model.layers)
    awq.check_mappings([n for n, _ in model.named_modules()], n_layers)

    train = [e for e in load_examples("data/processed/examples.jsonl") if e.split == "train"]
    random.Random(args.seed).shuffle(train)
    calib = train[: args.n_calib]
    collate = Collator(processor, args.min_pixels, args.max_pixels)

    def input_ids_for(i: int) -> list[int]:
        batch = collate([calib[i]])
        return batch["input_ids"][0][batch["attention_mask"][0].bool()].tolist()

    def data_collator(rows: list[dict]) -> dict:
        batch = collate([calib[row["i"]] for row in rows])
        batch.pop("labels")
        return dict(batch)

    oneshot(
        model=model,
        processor=processor,
        dataset=Dataset.from_dict(awq.calibration_rows(len(calib), input_ids_for)),
        recipe=awq.build_recipe(args.scheme),
        max_seq_length=args.max_seq,
        num_calibration_samples=len(calib),
        data_collator=data_collator,
        sequential_targets=["Qwen2_5_VLDecoderLayer"],
    )
    awq.check_quantized(model, n_layers)
    model.save_pretrained(args.out, save_compressed=True)
    processor.save_pretrained(args.out)

    size = sum(f.stat().st_size for f in args.out.glob("*.safetensors"))
    merged_size = sum(f.stat().st_size for f in merged.glob("*.safetensors"))
    summary = {
        "adapter": args.adapter,
        "scheme": args.scheme,
        "calibration_receipts": len(calib),
        "seconds": time.perf_counter() - t0,
        "checkpoint_gb": size / 1e9,
        "merged_bf16_gb": merged_size / 1e9,
    }
    (args.out / "quant_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
