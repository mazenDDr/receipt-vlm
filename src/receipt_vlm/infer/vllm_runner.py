"""The same prediction loop on vLLM, for checkpoints that need real 4-bit kernels (AWQ W4A16).

vLLM lives in its own conda env ("vllm") and is imported inside functions. The prompt, image cap, greedy
decoding and output parsing match the Hugging Face runner, so scores are comparable. One receipt per call, so
latency is per receipt like the other variants.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from typing import Any

from receipt_vlm.eval.parse import extract_json
from receipt_vlm.infer.runner import InferConfig
from receipt_vlm.prompts import build_messages
from receipt_vlm.schemas import Prediction, ReceiptExample


def load(cfg: InferConfig) -> tuple[Any, Any]:
    import os

    # WSL2: vLLM's GPU worker needs pinned memory, which it turns off on WSL by default. Measured on this
    # machine: pinning works, and host->GPU copies run at 13.9 GB/s pinned vs 7.4 GB/s pageable.
    os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")
    # flashinfer's sampler compiles CUDA code on first use and needs the curand headers. Greedy decoding
    # doesn't sample, so vLLM's own sampler gives the same outputs without compiling anything.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from transformers import AutoProcessor
    from vllm import LLM

    llm = LLM(
        model=cfg.model,
        max_model_len=cfg.max_model_len,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={"min_pixels": cfg.min_pixels, "max_pixels": cfg.max_pixels},
        seed=0,
    )
    processor = AutoProcessor.from_pretrained(cfg.model)
    return llm, processor


def is_truncated(finish_reason: str | None) -> bool:
    return finish_reason == "length"


def predict(
    examples: Sequence[ReceiptExample], cfg: InferConfig, llm: Any, processor: Any
) -> Iterator[Prediction]:
    from PIL import Image
    from vllm import SamplingParams

    params = SamplingParams(
        temperature=0.0, max_tokens=cfg.max_new_tokens, repetition_penalty=cfg.repetition_penalty, seed=0
    )
    prompt = processor.apply_chat_template(build_messages(), add_generation_prompt=True, tokenize=False)
    for example in examples:
        with Image.open(example.image_path) as image:
            image = image.convert("RGB")
        t0 = time.perf_counter()
        out = llm.generate(
            [{"prompt": prompt, "multi_modal_data": {"image": image}}], params, use_tqdm=False
        )[0]
        latency = time.perf_counter() - t0
        completion = out.outputs[0]
        yield Prediction(
            example_id=example.example_id,
            variant=cfg.variant,
            raw_output=completion.text,
            parsed=extract_json(completion.text),
            prompt_tokens=len(out.prompt_token_ids),
            output_tokens=len(completion.token_ids),
            latency_s=latency,
            peak_vram_mb=None,  # vLLM reserves memory up front (gpu_memory_utilization); measured separately
            truncated=is_truncated(completion.finish_reason),
        )
