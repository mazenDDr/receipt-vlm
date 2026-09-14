"""Per-split stats: receipts, fields per receipt, numeric vs. text share, image and target sizes."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from receipt_vlm.eval.fields import flatten, kind
from receipt_vlm.prompts import serialize_target
from receipt_vlm.schemas import ReceiptExample


def percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    p50, p95 = np.percentile(values, [50, 95])
    return {"min": float(min(values)), "p50": float(p50), "p95": float(p95), "max": float(max(values))}


def split_stats(
    examples: list[ReceiptExample], count_tokens: Callable[[str], int] | None = None
) -> dict[str, dict[str, object]]:
    """Stats per split. `count_tokens` (the model tokenizer) adds target lengths, to set max_new_tokens."""
    out: dict[str, dict[str, object]] = {}
    for split in ("train", "dev", "test"):
        rows = [e for e in examples if e.split == split]
        if not rows:
            continue
        kinds = [kind(path) for e in rows for path, _ in flatten(e.target)]
        keys = sorted({path for e in rows for path, _ in flatten(e.target)})
        stats: dict[str, object] = {
            "receipts": len(rows),
            "fields": len(kinds),
            "numeric_share": kinds.count("numeric") / len(kinds) if kinds else 0.0,
            "fields_per_receipt": percentiles([e.n_fields for e in rows]),
            "width": percentiles([e.width for e in rows]),
            "height": percentiles([e.height for e in rows]),
            "megapixels": percentiles([e.width * e.height / 1e6 for e in rows]),
            "key_paths": keys,
        }
        if count_tokens is not None:
            stats["target_tokens"] = percentiles([count_tokens(serialize_target(e.target)) for e in rows])
        out[split] = stats
    return out
