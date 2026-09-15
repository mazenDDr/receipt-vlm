"""AWQ W4A16 for the fine-tuned model: merge the adapter, then smooth and quantize the language model only.

llm-compressor's built-in Qwen2.5-VL mappings use unanchored names. Measured on the real module tree, its
up_proj -> down_proj mapping, and the gate/up balance layers, also match the 32 vision-encoder MLPs, which
are not quantized. The mappings here are anchored to the language model and checked before use.
llm-compressor is imported inside functions: it lives in the "quant" env only.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

LM = r"re:.*language_model\.layers\.\d+\."

# (smooth layer, balance layers) in llm-compressor's "re:" syntax
MAPPINGS: list[tuple[str, list[str]]] = [
    (
        LM + r"input_layernorm$",
        [LM + r"self_attn\.q_proj$", LM + r"self_attn\.k_proj$", LM + r"self_attn\.v_proj$"],
    ),
    (LM + r"self_attn\.v_proj$", [LM + r"self_attn\.o_proj$"]),
    (LM + r"post_attention_layernorm$", [LM + r"mlp\.gate_proj$", LM + r"mlp\.up_proj$"]),
    (LM + r"mlp\.up_proj$", [LM + r"mlp\.down_proj$"]),
]
IGNORE = ["lm_head", r"re:.*visual.*"]  # vision tower and projector stay bf16, as in the NF4 variants


def matches(pattern: str, names: Iterable[str]) -> list[str]:
    if pattern.startswith("re:"):
        rx = re.compile(pattern[3:])
        return [n for n in names if rx.fullmatch(n)]
    return [n for n in names if n == pattern]


def check_mappings(names: Sequence[str], n_layers: int) -> None:
    """Each smooth and balance pattern must hit one module per language layer and none in the vision tower."""
    for smooth, balance in MAPPINGS:
        for pattern in (smooth, *balance):
            hits = matches(pattern, names)
            vision = [n for n in hits if "visual" in n]
            if vision or len(hits) != n_layers:
                raise RuntimeError(
                    f"{pattern} matches {len(hits)} modules ({len(vision)} in the vision tower), "
                    f"expected {n_layers}"
                )


def check_quantized(model: Any, n_layers: int) -> None:
    """After quantization: all 7 linear layers of every language layer are quantized, nothing else is."""
    quantized = [n for n, m in model.named_modules() if getattr(m, "quantization_scheme", None) is not None]
    stray = [n for n in quantized if "language_model.layers." not in n]
    if stray or len(quantized) != n_layers * 7:
        raise RuntimeError(
            f"{len(quantized)} quantized modules, expected {n_layers * 7}; outside the LM: {stray[:5]}"
        )


def build_recipe(scheme: str = "W4A16_ASYM") -> list[Any]:
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.modifiers.transform.awq import AWQModifier
    from llmcompressor.modifiers.transform.awq.mappings import AWQMapping

    mappings = [AWQMapping(smooth_layer=s, balance_layers=b) for s, b in MAPPINGS]
    return [
        AWQModifier(mappings=mappings),
        QuantizationModifier(targets=["Linear"], scheme=scheme, ignore=IGNORE),
    ]


def merge_adapter(base: str, adapter: str | Path, out_dir: Path) -> None:
    """Fold a LoRA adapter into the bf16 base on the CPU and save it with the processor."""
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(base, dtype=torch.bfloat16, device_map="cpu")
    model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    model.save_pretrained(out_dir)
    AutoProcessor.from_pretrained(base).save_pretrained(out_dir)
