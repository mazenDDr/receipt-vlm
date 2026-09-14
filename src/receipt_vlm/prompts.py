"""The target text the model learns to write. Training and evaluation both use it, so they can't drift."""

from __future__ import annotations

import json
from typing import Any


def serialize_target(target: dict[str, Any]) -> str:
    """Compact JSON in CORD's key order, non-ASCII characters kept as they are."""
    return json.dumps(target, ensure_ascii=False, separators=(",", ":"))
