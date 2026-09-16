"""Build the label-audit set: receipts, their images, and where gold and the model disagree.

    python scripts/prepare_label_audit.py --run outputs/runs/<dev run> --out outputs/label_audit

Disagreements are computed with the same `eval/` flattening and normalization the scorer uses, so what
the audit shows is exactly what the metric counted. Images are downscaled for review, never re-encoded
into the dataset.

Selection is deliberately biased: the receipts where the model and the labels disagree are where the
labelling conventions and the gold errors both live. A sample of ordinary receipts is included as well,
and each row says which group it came from, so the audit cannot be mistaken for a random sample.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from receipt_vlm.data.build import load_examples
from receipt_vlm.eval import report
from receipt_vlm.eval.fields import flatten, kind
from receipt_vlm.eval.normalize import lenient, strict
from receipt_vlm.schemas import ExampleScore


def receipt_f1(score: ExampleScore) -> float:
    c = score.overall
    denominator = 2 * c.tp + c.fp + c.fn
    return 2 * c.tp / denominator if denominator else 1.0


def select(scores: list[ExampleScore], n_worst: int, n_sample: int) -> list[tuple[str, str]]:
    """The n_worst receipts by field F1, plus n_sample spread evenly through the rest."""
    ranked = sorted(scores, key=receipt_f1)
    worst = [(s.example_id, "worst") for s in ranked[:n_worst]]
    rest = ranked[n_worst:]
    if not rest or n_sample <= 0:
        return worst
    step = max(len(rest) // n_sample, 1)
    sample = [(s.example_id, "sample") for s in rest[::step][:n_sample]]
    return worst + sample


def by_key(pairs: list[tuple[str, str]]) -> dict[str, list[str]]:
    """Flattened (key path, value) pairs grouped by key, keeping repeats (menus have many items)."""
    grouped: dict[str, list[str]] = defaultdict(list)
    for path, value in pairs:
        grouped[path].append(value)
    return dict(grouped)


def compare(target: dict[str, Any], parsed: dict[str, Any] | None) -> list[dict[str, Any]]:
    """One row per key path: the gold values, the model's values, and whether they match.

    Two comparisons, because the audit exists to tell them apart:

    `agrees` is `normalize.strict`, Donut's strip-only rule and the one the headline metric counts. It
    treats "20,000" and "20.000" as different, which is the intended strictness when digits are the point.

    `agrees_lenient` applies `normalize.lenient` to numeric keys, dropping separators, currency and zero
    cents. A row that disagrees strictly but agrees leniently is a **labelling convention**, not a
    misreading - which is exactly what this audit has to separate out. (Using `strict` for both, as the
    first version of this script did, can never find one: the two comparisons are then the same test.)
    """
    gold_pairs = flatten(target)
    pred_pairs = flatten(parsed) if parsed else []
    gold, pred = by_key(gold_pairs), by_key(pred_pairs)
    rows = []
    for path in sorted(set(gold) | set(pred)):
        gold_values, pred_values = gold.get(path, []), pred.get(path, [])
        normalize = lenient if kind(path) == "numeric" else strict
        same_strict = Counter(strict(v) for v in gold_values) == Counter(strict(v) for v in pred_values)
        same_lenient = Counter(normalize(v) for v in gold_values) == Counter(
            normalize(v) for v in pred_values
        )
        rows.append(
            {
                "path": path,
                "kind": kind(path),
                "gold": gold_values,
                "pred": pred_values,
                "agrees": same_strict,
                "agrees_lenient": same_lenient,
            }
        )
    return rows


def save_image(source: Path, destination: Path, long_edge: int) -> tuple[int, int]:
    from PIL import Image

    with Image.open(source) as image:
        image = image.convert("RGB")
        scale = min(1.0, long_edge / max(image.size))
        if scale < 1.0:
            image = image.resize((round(image.width * scale), round(image.height * scale)))
        image.save(destination, "JPEG", quality=82, optimize=True)
        return image.size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="a scored run directory on the dev split")
    parser.add_argument("--examples-path", default="data/processed/examples.jsonl")
    parser.add_argument("--out", default="outputs/label_audit")
    parser.add_argument("--n-worst", type=int, default=20)
    parser.add_argument("--n-sample", type=int, default=10)
    parser.add_argument("--long-edge", type=int, default=1100, help="downscale for review")
    args = parser.parse_args()

    run = Path(args.run)
    scores = report.load_scores(run / "scores.jsonl")
    predictions = report.load_predictions(run / "predictions.jsonl")
    examples = {e.example_id: e for e in load_examples(args.examples_path)}

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    chosen = select(scores, args.n_worst, args.n_sample)
    scored = {s.example_id: s for s in scores}

    receipts = []
    for example_id, group in chosen:
        example, prediction = examples[example_id], predictions[example_id]
        name = f"{example_id}.jpg"
        width, height = save_image(Path(example.image_path), out / "images" / name, args.long_edge)
        rows = compare(example.target, prediction.parsed)
        receipts.append(
            {
                "example_id": example_id,
                "group": group,
                "f1": round(receipt_f1(scored[example_id]), 3),
                "image": f"images/{name}",
                "width": width,
                "height": height,
                "valid_json": prediction.parsed is not None,
                "rows": rows,
                "n_disagreements": sum(not r["agrees"] for r in rows),
                # A separator or currency convention rather than a misreading: strictly different,
                # leniently the same.
                "n_convention_only": sum(r["agrees_lenient"] and not r["agrees"] for r in rows),
                "gold": example.target,
                "pred": prediction.parsed,
            }
        )

    manifest = {
        "run": str(run),
        "variant": scores[0].variant if scores else "",
        "n_receipts": len(receipts),
        "receipts": receipts,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    total = sum(r["n_disagreements"] for r in receipts)
    print(f"{out}/manifest.json: {len(receipts)} receipts, {total} disagreeing key paths")
    print(f"{out}/images: {len(receipts)} JPEGs at {args.long_edge}px")


if __name__ == "__main__":
    main()
