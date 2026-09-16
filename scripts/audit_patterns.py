"""Count the labelling patterns found in the hand audit, across a whole split.

    python scripts/audit_patterns.py --run outputs/runs/<dev run>

The audit found six recurring differences between CORD's labels and the model. Each is stated as a rule
here and counted over every receipt in the run, so the guidelines can say how often each one happens
instead of how memorable it was.

Menu items are aligned by position, which is what the receipt itself implies: both sides list items in
printed order. An alignment is reported only where both sides have the same number of items, so a
nesting difference cannot masquerade as a per-item error.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from receipt_vlm.data.build import load_examples
from receipt_vlm.eval import report
from receipt_vlm.eval.fields import kind
from receipt_vlm.eval.normalize import lenient, strict

UNIT_COUNT = re.compile(r"^(\d+)\s*([A-Za-z]{1,6})\.?$")  # "1Prs", "2 x", "1 PCS"


def items(node: Any) -> list[dict[str, Any]]:
    """CORD writes a single menu item as an object and several as a list; normalize to a list."""
    if isinstance(node, list):
        return [i for i in node if isinstance(i, dict)]
    return [node] if isinstance(node, dict) else []


def has_sub(menu: list[dict[str, Any]]) -> bool:
    return any("sub" in item for item in menu)


def leaf_values(node: Any, path: str = "") -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            out += leaf_values(value, f"{path}.{key}" if path else key)
    elif isinstance(node, list):
        for value in node:
            out += leaf_values(value, path)
    else:
        out.append((path, str(node)))
    return out


def separator_direction(gold: str, pred: str) -> str | None:
    """Which way a thousands separator was rewritten, when only the separators differ."""
    if lenient(gold) != lenient(pred) or strict(gold) == strict(pred):
        return None
    if gold.count(",") > pred.count(",") and pred.count(".") > gold.count("."):
        return "comma_to_period"
    if gold.count(".") > pred.count(".") and pred.count(",") > gold.count(","):
        return "period_to_comma"
    return "other_formatting"


def analyse(target: dict[str, Any], parsed: dict[str, Any] | None) -> dict[str, list[str]]:
    """Every pattern this receipt exhibits, as short evidence strings."""
    found: dict[str, list[str]] = {}

    def note(name: str, evidence: str) -> None:
        found.setdefault(name, []).append(evidence)

    if parsed is None:
        note("invalid_json", "output did not parse")
        return found

    gold_menu, pred_menu = items(target.get("menu")), items(parsed.get("menu"))

    # 1. The model nests sub-items the labels keep flat (or the reverse).
    if has_sub(pred_menu) and not has_sub(gold_menu):
        note("model_added_sub", f"{len(pred_menu)} items, sub nesting not in the label")
    if has_sub(gold_menu) and not has_sub(pred_menu):
        note("model_flattened_sub", f"label nests, prediction is flat ({len(pred_menu)} items)")

    # 2. Thousands separators rewritten.
    gold_leaves, pred_leaves = dict(leaf_values(target)), dict(leaf_values(parsed))
    for path, gold_value in leaf_values(target):
        if kind(path) != "numeric":
            continue
        pred_value = pred_leaves.get(path)
        if not isinstance(pred_value, str):
            continue
        direction = separator_direction(gold_value, pred_value)
        if direction:
            note(direction, f"{path}: {gold_value!r} -> {pred_value!r}")

    # Items are comparable only when both sides list the same number of them.
    if len(gold_menu) == len(pred_menu):
        for gold_item, pred_item in zip(gold_menu, pred_menu, strict=True):
            gold_cnt, pred_cnt = str(gold_item.get("cnt", "")), str(pred_item.get("cnt", ""))
            gold_nm, pred_nm = str(gold_item.get("nm", "")), str(pred_item.get("nm", ""))

            # 3. The unit travels from the count to the front of the name.
            unit = UNIT_COUNT.match(gold_cnt)
            if unit and pred_cnt.strip().isdigit() and unit.group(2).lower() in pred_nm.lower():
                note("unit_moved_to_name", f"cnt {gold_cnt!r} -> {pred_cnt!r}, nm {pred_nm!r}")

            # 4. A count printed beside the name, which the model does not pick up.
            if gold_cnt and not pred_cnt:
                note("count_missed", f"label cnt {gold_cnt!r}, nothing predicted (nm {gold_nm!r})")

            # 6. Spaces inserted into a name that is printed closed up.
            same_closed_up = gold_nm.replace(" ", "") == pred_nm.replace(" ", "")
            if gold_nm and pred_nm and gold_nm != pred_nm and same_closed_up:
                longer = "model" if pred_nm.count(" ") > gold_nm.count(" ") else "label"
                note("name_spacing", f"{gold_nm!r} -> {pred_nm!r} ({longer} has more spaces)")

    # 5. A total or subtotal value filed under a different section key.
    for section in ("sub_total", "total"):
        gold_section = target.get(section) or {}
        pred_section = parsed.get(section) or {}
        if not isinstance(gold_section, dict) or not isinstance(pred_section, dict):
            continue
        for gold_key, gold_value in gold_section.items():
            if gold_key in pred_section:
                continue
            moved = [k for k, v in pred_section.items() if k not in gold_section and v == gold_value]
            if moved:
                note("value_filed_under_another_key", f"{section}.{gold_key} -> {section}.{moved[0]}")

    _ = gold_leaves  # kept for symmetry with pred_leaves above
    return found


DESCRIPTIONS = {
    "model_added_sub": "model nests sub-items; the label keeps them flat",
    "model_flattened_sub": "label nests sub-items; the model lists them flat",
    "comma_to_period": "thousands separator read as '.' where the label has ','",
    "period_to_comma": "thousands separator read as ',' where the label has '.'",
    "other_formatting": "same digits, different punctuation or spacing",
    "unit_moved_to_name": "unit moved from the count onto the front of the name",
    "count_missed": "label has a count, the model predicted none",
    "name_spacing": "same name, different spacing",
    "value_filed_under_another_key": "right value, different section key",
    "invalid_json": "output did not parse",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--examples-path", default="data/processed/examples.jsonl")
    parser.add_argument("--out", default="")
    parser.add_argument("--examples-per-pattern", type=int, default=3)
    args = parser.parse_args()

    run = Path(args.run)
    predictions = report.load_predictions(run / "predictions.jsonl")
    examples = {e.example_id: e for e in load_examples(args.examples_path)}

    receipts = Counter()
    occurrences = Counter()
    evidence: dict[str, list[str]] = {}
    scored = 0
    for example_id, prediction in predictions.items():
        example = examples.get(example_id)
        if example is None:
            continue
        scored += 1
        for name, hits in analyse(example.target, prediction.parsed).items():
            receipts[name] += 1
            occurrences[name] += len(hits)
            evidence.setdefault(name, []).extend(f"{example_id}  {h}" for h in hits)

    print(f"{scored} receipts from {run.name}\n")
    print(f"{'pattern':<32} {'receipts':>8} {'%':>5} {'fields':>7}  description")
    for name, count in receipts.most_common():
        print(
            f"{name:<32} {count:>8} {count / scored:>5.0%} {occurrences[name]:>7}  "
            f"{DESCRIPTIONS.get(name, '')}"
        )
    print()
    for name, _ in receipts.most_common():
        print(f"--- {name} ---")
        for line in evidence[name][: args.examples_per_pattern]:
            print(f"    {line}")

    if args.out:
        payload = {
            "run": str(run),
            "n_receipts": scored,
            "patterns": {
                name: {
                    "receipts": count,
                    "share": round(count / scored, 3),
                    "fields": occurrences[name],
                    "description": DESCRIPTIONS.get(name, ""),
                    "examples": evidence[name][:8],
                }
                for name, count in receipts.most_common()
            },
        }
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
