"""Paired comparison of two scored runs on the same receipts: each metric's difference with a 95% interval.

A difference counts as real only if its interval excludes 0.
"""

from __future__ import annotations

from pathlib import Path

from receipt_vlm.eval import metrics
from receipt_vlm.eval.bootstrap import paired_difference
from receipt_vlm.eval.report import LABELS, load_scores

Diff = tuple[float, float, float]  # (a - b, lo, hi)


def compare(a_dir: Path, b_dir: Path, n_boot: int = 2000, seed: int = 0) -> dict[str, Diff]:
    a, b = load_scores(a_dir / "scores.jsonl"), load_scores(b_dir / "scores.jsonl")
    return {
        name: paired_difference(a, b, metric, key=lambda s: s.example_id, n_boot=n_boot, seed=seed)
        for name, metric in metrics.METRICS.items()
    }


def is_real(diff: Diff) -> bool:
    _, lo, hi = diff
    return lo > 0 or hi < 0


def render(a_name: str, b_name: str, diffs: dict[str, Diff], n: int) -> str:
    lines = [
        f"# {a_name} minus {b_name} ({n} receipts, paired)",
        "",
        "| Metric | Difference | 95% interval | Real? |",
        "|---|---|---|---|",
    ]
    for name, (d, lo, hi) in diffs.items():
        real = "yes" if is_real((d, lo, hi)) else "no"
        lines.append(f"| {LABELS.get(name, name)} | {d:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {real} |")
    return "\n".join(lines) + "\n"
