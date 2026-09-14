"""The extraction prompt and the target text. Training and evaluation both use these, so they can't drift."""

from __future__ import annotations

import json
from typing import Any

# One prompt for every variant (zero-shot, fine-tuned, quantized). It names every CORD key, so the base model
# gets a fair zero-shot chance; tests check it stays in sync with eval/fields.py.
INSTRUCTION = """Read this receipt and return its contents as one JSON object.
Use only these keys, and leave out any that are not printed on the receipt:
- "menu": a list of purchased items. Item keys: "nm" (name), "num" (item code), "unitprice", \
"cnt" (quantity), "discountprice", "price" (line total), "itemsubtotal", "vatyn", "etc", and "sub" \
(a list of options or sub-items with the same keys).
- "void_menu": cancelled items, with "nm" and "price".
- "sub_total": "subtotal_price", "discount_price", "service_price", "othersvc_price", "tax_price", "etc".
- "total": "total_price", "total_etc", "cashprice", "changeprice", "creditcardprice", "emoneyprice", \
"menutype_cnt" (number of item types), "menuqty_cnt" (number of items).
Every value is a string copied exactly as printed, including separators and symbols (e.g. "16,500", "@8.000").
Answer with the JSON only."""


def serialize_target(target: dict[str, Any]) -> str:
    """Compact JSON in CORD's key order, non-ASCII characters kept as they are."""
    return json.dumps(target, ensure_ascii=False, separators=(",", ":"))


def build_messages(target: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Chat messages for one receipt image; with a target, the assistant turn the model learns to write."""
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": INSTRUCTION}]}
    ]
    if target is not None:
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": serialize_target(target)}]}
        )
    return messages
