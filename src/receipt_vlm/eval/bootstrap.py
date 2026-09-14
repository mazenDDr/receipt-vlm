"""Percentile bootstrap over receipts, and paired differences between two variants on the same receipts."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

import numpy as np

T = TypeVar("T")


def interval(
    items: Sequence[T],
    metric: Callable[[Sequence[T]], float],
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """(value, lo, hi): the metric on all items and its (1 - alpha) interval from resampling items."""
    rng = np.random.default_rng(seed)
    n = len(items)
    stats = [metric([items[i] for i in rng.integers(0, n, n)]) for _ in range(n_boot)]
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return metric(items), float(lo), float(hi)


def paired_difference(
    a: Sequence[T],
    b: Sequence[T],
    metric: Callable[[Sequence[T]], float],
    key: Callable[[T], str],
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """(diff, lo, hi) for metric(a) - metric(b), resampling the same receipts on both sides.

    The difference is real only if [lo, hi] excludes 0.
    """
    b_by_key = {key(item): item for item in b}
    if sorted(b_by_key) != sorted(key(item) for item in a):
        raise ValueError("paired comparison needs the same receipts on both sides")
    pairs = [(item, b_by_key[key(item)]) for item in a]
    rng = np.random.default_rng(seed)
    n = len(pairs)
    diffs = []
    for _ in range(n_boot):
        sample = [pairs[i] for i in rng.integers(0, n, n)]
        diffs.append(metric([x for x, _ in sample]) - metric([y for _, y in sample]))
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return metric([x for x, _ in pairs]) - metric([y for _, y in pairs]), float(lo), float(hi)
