"""Leakage check: the same receipt photo (exact bytes or a near copy) must not appear in two splits."""

from __future__ import annotations

from itertools import combinations

from PIL import Image
from pydantic import BaseModel

from receipt_vlm.schemas import ReceiptExample


class Pair(BaseModel):
    a: str
    b: str
    splits: tuple[str, str]
    distance: int  # Hamming distance between difference hashes; 0 for identical bytes
    exact: bool


def dhash(image: Image.Image, size: int = 16) -> int:
    """Difference hash: each pixel vs. its right neighbour on a (size+1) x size grayscale thumbnail."""
    gray = image.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    px = gray.tobytes()  # one byte per pixel in mode "L"
    bits = 0
    for row in range(size):
        for col in range(size):
            left, right = px[row * (size + 1) + col], px[row * (size + 1) + col + 1]
            bits = (bits << 1) | (left > right)
    return bits


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def nearest_pairs(examples: list[ReceiptExample], hashes: dict[str, int], max_distance: int) -> list[Pair]:
    """All pairs from different splits that are identical or within max_distance, closest first."""
    pairs = []
    for x, y in combinations(examples, 2):
        if x.split == y.split:
            continue
        exact = x.image_sha256 == y.image_sha256
        distance = 0 if exact else hamming(hashes[x.example_id], hashes[y.example_id])
        if exact or distance <= max_distance:
            pairs.append(
                Pair(
                    a=x.example_id, b=y.example_id, splits=(x.split, y.split), distance=distance, exact=exact
                )
            )
    return sorted(pairs, key=lambda p: (p.distance, p.a, p.b))


def nearest_distance(examples: list[ReceiptExample], hashes: dict[str, int]) -> dict[str, int]:
    """Per dev/test receipt: distance to its closest receipt in another split (the leakage margin)."""
    out = {}
    for x in examples:
        if x.split == "train":
            continue
        others = [hamming(hashes[x.example_id], hashes[y.example_id]) for y in examples if y.split != x.split]
        out[x.example_id] = min(others, default=-1)
    return out


_SPLIT_ORDER = {"train": 0, "dev": 1, "test": 2}


def exclusions(pairs: list[Pair]) -> dict[str, Pair]:
    """Which receipt of each cross-split copy to drop: the train copy, or the dev copy of a dev/test pair.

    The official test split stays whole, so results stay comparable with published CORD numbers.
    """
    out: dict[str, Pair] = {}
    for p in pairs:
        drop = p.a if _SPLIT_ORDER[p.splits[0]] < _SPLIT_ORDER[p.splits[1]] else p.b
        out.setdefault(drop, p)
    return out
