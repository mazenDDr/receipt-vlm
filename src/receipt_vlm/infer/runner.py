"""Greedy generation with the Hugging Face model (bf16 or NF4, optionally with a LoRA adapter).

Every receipt yields a Prediction: raw output, parsed JSON, token counts, wall-clock latency and peak VRAM.
torch and transformers are imported inside functions, so the helpers and tests run on a machine without a GPU.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from typing import Any, Literal

from pydantic import BaseModel

from receipt_vlm.eval.parse import extract_json
from receipt_vlm.prompts import build_messages
from receipt_vlm.schemas import Prediction, ReceiptExample


class InferConfig(BaseModel):
    model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    adapter: str | None = None  # LoRA adapter dir for fine-tuned variants
    precision: Literal["bf16", "nf4"] = "bf16"
    variant: str = "base-bf16"
    split: Literal["train", "dev", "test"] = "dev"
    examples_path: str = "data/processed/examples.jsonl"
    min_pixels: int = 56 * 56
    max_pixels: int = 1024 * 28 * 28
    max_new_tokens: int = 1024
    # Qwen's generation_config sets 1.05, which pushes the model away from repeating "1", "0", ".000" ...
    repetition_penalty: float = 1.0
    batch_size: int = 1
    limit: int | None = None
    attn_implementation: str = "sdpa"


def image_kwargs(cfg: InferConfig) -> dict[str, Any]:
    # transformers 5.8 silently ignores `max_pixels` and a load-time `size=`; only this per-call form works.
    return {"size": {"shortest_edge": cfg.min_pixels, "longest_edge": cfg.max_pixels}}


def check_image_cap(grid: Sequence[Sequence[int]], patch_size: int, max_pixels: int) -> None:
    """Fail loudly if the processor didn't apply the pixel cap (grid rows are [t, h, w] in patches)."""
    worst = max(int(h) * int(w) * patch_size**2 for _, h, w in grid)
    if worst > max_pixels:
        raise RuntimeError(f"image resized to {worst} pixels, above max_pixels={max_pixels}; cap was ignored")


def generated_tokens(ids: Sequence[int], eos_ids: set[int]) -> list[int]:
    """The generated ids up to and including the first end-of-sequence token (the rest is padding)."""
    for i, token in enumerate(ids):
        if token in eos_ids:
            return list(ids[: i + 1])
    return list(ids)


def load(cfg: InferConfig) -> tuple[Any, Any]:
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    quantization = None
    if cfg.precision == "nf4":
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg.model,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation=cfg.attn_implementation,
        quantization_config=quantization,
    )
    if cfg.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, cfg.adapter)
    model.eval()
    processor = AutoProcessor.from_pretrained(cfg.model)
    processor.tokenizer.padding_side = "left"  # generation continues from the right edge
    return model, processor


def predict(
    examples: Sequence[ReceiptExample], cfg: InferConfig, model: Any, processor: Any
) -> Iterator[Prediction]:
    import torch
    from PIL import Image

    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    prompt = processor.apply_chat_template(build_messages(), add_generation_prompt=True, tokenize=False)
    for start in range(0, len(examples), cfg.batch_size):
        batch = examples[start : start + cfg.batch_size]
        images = []
        for example in batch:
            with Image.open(example.image_path) as image:
                images.append(image.convert("RGB"))
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        enc = processor(
            text=[prompt] * len(batch),
            images=images,
            return_tensors="pt",
            padding=True,
            images_kwargs=image_kwargs(cfg),
        )
        check_image_cap(enc["image_grid_thw"].tolist(), processor.image_processor.patch_size, cfg.max_pixels)
        enc = enc.to(model.device)
        with torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                repetition_penalty=cfg.repetition_penalty,
            )
        torch.cuda.synchronize()
        latency = (time.perf_counter() - t0) / len(batch)  # batch wall time shared by its receipts
        peak_mb = torch.cuda.max_memory_allocated() / 2**20
        new = out[:, enc["input_ids"].shape[1] :].tolist()
        for example, ids, mask in zip(batch, new, enc["attention_mask"].tolist(), strict=True):
            tokens = generated_tokens(ids, eos_ids)
            raw = processor.tokenizer.decode(tokens, skip_special_tokens=True)
            yield Prediction(
                example_id=example.example_id,
                variant=cfg.variant,
                raw_output=raw,
                parsed=extract_json(raw),
                prompt_tokens=int(sum(mask)),
                output_tokens=len(tokens),
                latency_s=latency,
                peak_vram_mb=peak_mb,
                truncated=not any(t in eos_ids for t in tokens),
            )
