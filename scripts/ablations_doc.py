"""Write docs/ablations.md from the dev runs: every run's scores and each ablation's paired difference.

    python scripts/ablations_doc.py      # needs each run's scores.jsonl locally (./gpu pull)

Every number in the page comes from the run folders; nothing is typed by hand.
"""

from __future__ import annotations

from pathlib import Path

from receipt_vlm.eval import compare, tables

RUNS = Path("outputs/runs")
TABLE = [
    "base-bf16",
    "qlora-r16-lm-lr1e4",
    "qlora-r16-lm",
    "qlora-r16-lm-lr4e4",
    "qlora-r16-lm-lr4e4-e1",
    "qlora-r8-lm-lr4e4-e1",
    "qlora-r32-lm-lr4e4-e1",
    "qlora-r16-lmproj-lr4e4-e1",
    "lora-bf16-r16-lm-lr4e4-e1",
    "partial4-bf16-lr1e5-e1",
    "partial4-bf16-lr5e5-e1",
]
# (question, arm, reference)
COMPARISONS = [
    ("Fine-tuning at all (QLoRA r16, lr 2e-4, 2 epochs)", "qlora-r16-lm", "base-bf16"),
    ("A. Learning rate 1e-4 instead of 2e-4", "qlora-r16-lm-lr1e4", "qlora-r16-lm"),
    ("A. Learning rate 4e-4 instead of 2e-4", "qlora-r16-lm-lr4e4", "qlora-r16-lm"),
    ("One epoch instead of two (r16, lr 4e-4)", "qlora-r16-lm-lr4e4-e1", "qlora-r16-lm-lr4e4"),
    ("B. Rank 8 instead of 16", "qlora-r8-lm-lr4e4-e1", "qlora-r16-lm-lr4e4-e1"),
    ("B. Rank 32 instead of 16", "qlora-r32-lm-lr4e4-e1", "qlora-r16-lm-lr4e4-e1"),
    ("C. Also adapt the vision projector", "qlora-r16-lmproj-lr4e4-e1", "qlora-r16-lm-lr4e4-e1"),
    ("D. LoRA on a bf16 base instead of 4-bit (QLoRA)", "lora-bf16-r16-lm-lr4e4-e1", "qlora-r16-lm-lr4e4-e1"),
    ("E. Partial full fine-tuning, lr 1e-5", "partial4-bf16-lr1e5-e1", "qlora-r16-lm-lr4e4-e1"),
    ("E. Partial full fine-tuning, lr 5e-5", "partial4-bf16-lr5e5-e1", "qlora-r16-lm-lr4e4-e1"),
]


def run_dir(variant: str) -> Path:
    pattern = "*_base-bf16_dev" if variant == "base-bf16" else f"*_{variant}_train"
    matches = sorted(RUNS.glob(pattern))
    if not matches:
        raise SystemExit(f"no run folder for {variant} ({pattern})")
    return matches[-1]


def _cell(diff: compare.Diff) -> str:
    d, lo, hi = diff
    return f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def main() -> None:
    lines = [
        "# Ablations",
        "",
        "Dev split (99 CORD v2 receipts). Each fine-tuned arm is scored by generating the JSON and",
        "comparing it field by field with the labels. Differences are **paired**: both arms are scored",
        "on the same receipts, and the 95% interval comes from 2,000 bootstrap resamples of those",
        "receipts. A difference counts as real only if its interval excludes 0. From the rank stage on,",
        "each arm trains one epoch and is compared with a one-epoch reference, so every comparison is",
        "like for like.",
        "",
        "## Every run",
        "",
        tables.render([tables.run_row(run_dir(v)) for v in TABLE]).rstrip(),
        "",
        "## What each choice changes",
        "",
        "| Question | Field F1 difference [95%] | Real? | Numeric F1 | Text F1 |",
        "|---|---|---|---|---|",
    ]
    for question, arm, reference in COMPARISONS:
        diffs = compare.compare(run_dir(arm), run_dir(reference))
        real = "yes" if compare.is_real(diffs["f1"]) else "no"
        lines.append(
            f"| {question} | {_cell(diffs['f1'])} | {real} | "
            f"{_cell(diffs['f1_numeric'])} | {_cell(diffs['f1_text'])} |"
        )
    lines += [
        "",
        'Rows marked "no" are ties within noise, not evidence that the choice doesn\'t matter at all.',
        "",
    ]
    Path("docs/ablations.md").write_text("\n".join(lines))
    print(f"wrote docs/ablations.md ({len(TABLE)} runs, {len(COMPARISONS)} comparisons)")


if __name__ == "__main__":
    main()
