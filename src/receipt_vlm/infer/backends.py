"""Which prediction loop a variant uses: transformers (default) or vLLM (AWQ W4A16 checkpoints)."""

from __future__ import annotations

from types import ModuleType

from receipt_vlm.infer import llamacpp_runner, runner, vllm_runner


def get(name: str) -> ModuleType:
    if name == "hf":
        return runner
    if name == "vllm":
        return vllm_runner
    if name == "llamacpp":
        return llamacpp_runner
    raise ValueError(f"unknown backend {name!r}")
