"""Turning a raw model output into the receipt dict that gets scored."""

from __future__ import annotations

import json
from typing import Any


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def _merge_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    # A model may repeat "menu": {...} per item instead of a list; json.loads would keep only the last one.
    out: dict[str, Any] = {}
    for key, value in pairs:
        out[key] = [*_as_list(out[key]), *_as_list(value)] if key in out else value
    return out


def extract_json(raw: str) -> dict[str, Any] | None:
    """The receipt dict in a model output, or None.

    Tolerated: code fences or text around one JSON object, and repeated keys (merged into a list).
    Not repaired: truncated or malformed JSON, because that is a real failure of the model.
    """
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end < start:
        return None
    try:
        obj = json.loads(raw[start : end + 1], object_pairs_hook=_merge_duplicate_keys)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None
