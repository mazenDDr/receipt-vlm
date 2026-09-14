"""Shared data contracts. Every intermediate file is JSONL with one of these models per line."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

Split = Literal["train", "dev", "test"]


class ReceiptExample(BaseModel):
    """One receipt image and its gold fields."""

    example_id: str  # f"cord-{cord_split}-{row:04d}", stable across rebuilds
    split: Split  # CORD validation is our dev split
    image_path: str  # relative to the project root, e.g. "data/processed/images/cord-test-0000.png"
    image_sha256: str
    width: int
    height: int
    target: dict[str, Any]  # CORD gt_parse, whitespace-normalized, otherwise unchanged
    n_fields: int  # leaf (key, value) pairs in target


class Prediction(BaseModel):
    """One model output for one receipt."""

    example_id: str
    variant: str  # e.g. "base-bf16", "ft-r16-bf16", "ft-r16-awq-w4a16", "ft-r16-gguf-q4_k_m"
    raw_output: str
    parsed: dict[str, Any] | None  # None when the output isn't valid JSON
    prompt_tokens: int
    output_tokens: int
    latency_s: float  # wall-clock: image preprocessing + generation
    peak_vram_mb: float | None = None
    truncated: bool = False  # stopped at max_new_tokens


class FieldCounts(BaseModel):
    tp: int = 0
    fp: int = 0
    fn: int = 0


class ExampleScore(BaseModel):
    """Field-level counts for one prediction. F1 comes from counts summed over receipts, not averaged."""

    example_id: str
    variant: str
    valid_json: bool
    exact_match: bool
    overall: FieldCounts
    numeric: FieldCounts
    text: FieldCounts
    ted_accuracy: float | None = None
