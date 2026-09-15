"""A table of scored runs on one split: accuracy with intervals, plus training cost for trained runs."""

from __future__ import annotations

import json
from pathlib import Path

METRIC_KEYS = ("f1", "f1_numeric", "f1_text", "precision", "recall", "valid_json", "ted_accuracy")


def run_row(run_dir: Path) -> dict[str, object]:
    summary = json.loads((run_dir / "summary.json").read_text())
    train_path = run_dir / "train_summary.json"
    train = json.loads(train_path.read_text()) if train_path.exists() else None
    return {
        "run": run_dir.name,
        "variant": summary["variant"],
        "split": summary["split"],
        "n": summary["n"],
        **{k: summary["metrics"][k] for k in METRIC_KEYS},
        "train_minutes": train["train_seconds"] / 60 if train else None,
        # null when the job stopped before recording it; shown as "—", never as 0
        "train_peak_gb": train["train_peak_vram_mb"] / 1024
        if train and train["train_peak_vram_mb"]
        else None,
        "trainable_m": train["trainable_params"] / 1e6 if train else None,
    }


def check_same_split(rows: list[dict[str, object]]) -> None:
    splits = sorted({str(r["split"]) for r in rows})
    if len(splits) > 1:
        raise ValueError(f"runs from different splits {splits}; put one split in a table")


def _num(value: object, fmt: str) -> str:
    return "—" if value is None else format(value, fmt)


def render(rows: list[dict[str, object]]) -> str:
    check_same_split(rows)
    lines = [
        "| Variant | n | Field F1 [95% interval] | Numeric F1 | Text F1 | Precision | Recall "
        "| Valid JSON | TED | Train min | Train peak GB | Trainable M |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        f1 = r["f1"]
        cells = [
            str(r["variant"]),
            str(r["n"]),
            f"{f1['value']:.3f} [{f1['lo']:.3f}, {f1['hi']:.3f}]",
            *(f"{r[k]['value']:.3f}" for k in METRIC_KEYS[1:]),
            _num(r["train_minutes"], ".0f"),
            _num(r["train_peak_gb"], ".1f"),
            _num(r["trainable_m"], ".1f"),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
