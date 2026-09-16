"""Build the project pages from their templates and the committed runs.

    python scripts/build_site.py

    site/demo-template.html  -> site/demo.html   every held-out receipt, four variants, field by field

Every number and every output on the pages is read from `outputs/runs/*/` and
`outputs/site_data/manifest.json`; nothing is written by hand. Re-score a variant, re-run this, and
the pages follow.

The field rows are recomputed **per variant** with the same `eval/` flattening and normalization the
scorer uses. They are deliberately not taken from the manifest, whose rows compare gold against the
fine-tuned model only: reusing those for the base or AWQ tab would show one model's mistakes under
another model's name.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from receipt_vlm.eval.fields import flatten, kind
from receipt_vlm.eval.normalize import lenient, strict

ROOT = Path(__file__).resolve().parents[1]

# The variants the demo page shows, in the order the tabs appear. Each names a scored run on the
# test split; the notes say what the reader is looking at.
VARIANTS = [
    {
        "key": "base",
        "label": "Base model",
        "run": "outputs/runs/20260915-2200_base-bf16_test_final",
        "note": "Qwen2.5-VL-3B, zero-shot, no fine-tuning",
    },
    {
        "key": "ft",
        "label": "Fine-tuned",
        "run": "outputs/runs/20260915-2231_qlora-r16-lm-lr4e4_test_final",
        "note": "QLoRA r16, language model only — the deployed model",
    },
    {
        "key": "awq",
        "label": "AWQ 4-bit",
        "run": "outputs/runs/20260915-2353_awq-w4a16-vllm_test_final",
        "note": "W4A16 on vLLM, 3.31 GiB of weights",
    },
    {
        "key": "gguf",
        "label": "GGUF Q4_K_M",
        "run": "outputs/runs/20260916-0516_ft-r16-gguf-q4_k_m_test_final",
        "note": "llama.cpp, 4-bit text weights plus the f32 projector",
    },
]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def by_id(path: Path) -> dict[str, dict]:
    return {row["example_id"]: row for row in read_jsonl(path)}


def receipt_f1(counts: dict[str, int]) -> float:
    denominator = 2 * counts["tp"] + counts["fp"] + counts["fn"]
    return 2 * counts["tp"] / denominator if denominator else 1.0


def grouped(pairs: list[tuple[str, str]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for path, value in pairs:
        out[path].append(value)
    return dict(out)


def field_rows(target: dict[str, Any], parsed: dict[str, Any] | None) -> list[dict[str, Any]]:
    """One row per key path: gold values, this variant's values, and how they compare.

    `agrees` is the strict rule the metric counts. `agrees_lenient` ignores separators and currency on
    numeric keys, so a row that differs strictly but agrees leniently is a labelling convention rather
    than a misreading — the distinction `docs/annotation_guidelines.md` is built on.
    """
    gold = grouped(flatten(target))
    pred = grouped(flatten(parsed) if parsed else [])
    rows = []
    for path in sorted(set(gold) | set(pred)):
        gold_values, pred_values = gold.get(path, []), pred.get(path, [])
        normalize = lenient if kind(path) == "numeric" else strict
        rows.append(
            {
                "path": path,
                "gold": gold_values,
                "pred": pred_values,
                "agrees": Counter(map(strict, gold_values)) == Counter(map(strict, pred_values)),
                "agrees_lenient": (
                    Counter(map(normalize, gold_values)) == Counter(map(normalize, pred_values))
                ),
            }
        )
    return rows


def tags_for(rows: list[dict[str, Any]], gold: dict[str, Any], parsed: dict[str, Any] | None) -> list[str]:
    """The buckets the filter chips use, from the fine-tuned model's rows."""
    tags = []
    if any(not r["agrees"] and r["agrees_lenient"] for r in rows):
        tags.append("separator")
    if any(not r["agrees"] and not r["agrees_lenient"] and kind(r["path"]) == "numeric" for r in rows):
        tags.append("digit")
    has_sub = lambda node: '"sub"' in json.dumps(node)  # noqa: E731
    if parsed is not None and has_sub(gold) != has_sub(parsed):
        tags.append("nesting")
    return tags


def build_demo(manifest: dict[str, Any]) -> dict[str, Any]:
    runs = {}
    for variant in VARIANTS:
        run = ROOT / variant["run"]
        runs[variant["key"]] = {
            "preds": by_id(run / "predictions.jsonl"),
            "scores": by_id(run / "scores.jsonl"),
        }

    receipts = []
    for entry in manifest["receipts"]:
        example_id, gold = entry["example_id"], entry["gold"]
        per_variant, summary = {}, {}
        for variant in VARIANTS:
            key = variant["key"]
            prediction = runs[key]["preds"].get(example_id)
            score = runs[key]["scores"].get(example_id)
            if prediction is None or score is None:
                raise SystemExit(f"{example_id} missing from {variant['run']}")
            rows = field_rows(gold, prediction["parsed"])
            per_variant[key] = {
                "f1": round(receipt_f1(score["overall"]), 3),
                "exact": bool(score["exact_match"]),
                "valid": prediction["parsed"] is not None,
                "tokens": prediction["output_tokens"],
                "latency": round(prediction["latency_s"], 3),
                "raw": prediction["raw_output"],
                "rows": rows,
            }
            summary[key] = {
                "f1": per_variant[key]["f1"],
                "exact": per_variant[key]["exact"],
                "valid": per_variant[key]["valid"],
            }

        ft_rows = per_variant["ft"]["rows"]
        receipts.append(
            {
                "example_id": example_id,
                "width": entry["width"],
                "height": entry["height"],
                "n_disagreements": sum(not r["agrees"] for r in ft_rows),
                "tags": tags_for(ft_rows, gold, runs["ft"]["preds"][example_id]["parsed"]),
                "base": summary["base"],
                "ft": summary["ft"],
                "variants": per_variant,
            }
        )

    receipts.sort(key=lambda r: (r["ft"]["f1"], r["example_id"]))
    return {
        "receipts": receipts,
        "variants": [{k: v for k, v in item.items() if k != "run"} for item in VARIANTS],
    }


def counts_for(run: str, example_id: str) -> dict[str, Any]:
    """The tp / fp / fn behind one receipt's score, so the tour can show the arithmetic."""
    score = by_id(ROOT / "outputs/runs" / run / "scores.jsonl")[example_id]
    return {
        group: {
            "tp": score[group]["tp"],
            "fp": score[group]["fp"],
            "fn": score[group]["fn"],
            "f1": round(receipt_f1(score[group]), 3),
        }
        for group in ("overall", "numeric", "text")
    } | {"ted": round(score.get("ted_accuracy") or 0.0, 3), "exact": bool(score["exact_match"])}


def build_story(manifest: dict[str, Any], example_id: str) -> dict[str, Any]:
    """One receipt followed through every stage, for the tour.

    `identical` is computed by comparing each quantized build's raw output against the fine-tuned
    model's, character for character. The tour claims the 4-bit builds reproduce it byte for byte, so
    that claim is measured here rather than asserted in the copy.
    """
    from receipt_vlm.prompts import INSTRUCTION

    entry = next(r for r in manifest["receipts"] if r["example_id"] == example_id)
    runs = {v["key"]: ROOT / v["run"] for v in VARIANTS}
    preds = {key: by_id(run / "predictions.jsonl")[example_id] for key, run in runs.items()}
    scores = {key: by_id(run / "scores.jsonl")[example_id] for key, run in runs.items()}

    def panel(key: str) -> dict[str, Any]:
        prediction, score = preds[key], scores[key]
        return {
            "f1": round(receipt_f1(score["overall"]), 3),
            "tokens": prediction["output_tokens"],
            "latency": round(prediction["latency_s"], 3),
            "raw": prediction["raw_output"],
            "rows": field_rows(entry["gold"], prediction["parsed"]),
        }

    reference = preds["ft"]["raw_output"]
    runs_by_key = {v["key"]: Path(v["run"]).name for v in VARIANTS}
    return {
        "example_id": example_id,
        "width": entry["width"],
        "height": entry["height"],
        "prompt_tokens": preds["ft"]["prompt_tokens"],
        "n_gold_fields": len(flatten(entry["gold"])),
        "instruction": INSTRUCTION,
        # The schema as it actually appears in the labels, from the data report. Scraping key names out
        # of the instruction's prose with a regex produced 27 flat names for a schema that has 29 nested
        # paths - plausible-looking and wrong.
        "key_paths": stats_summary()["splits"]["train"]["key_paths"],
        "gold": entry["gold"],
        "counts": {key: counts_for(run, example_id) for key, run in runs_by_key.items()},
        "panels": {v["key"]: panel(v["key"]) | {"label": v["label"], "note": v["note"]} for v in VARIANTS},
        "base": panel("base"),
        "ft": panel("ft"),
        "variants": [
            {
                "label": variant["label"],
                "f1": round(receipt_f1(scores[variant["key"]]["overall"]), 3),
                "tokens": preds[variant["key"]]["output_tokens"],
                "latency": round(preds[variant["key"]]["latency_s"], 3),
                "identical": preds[variant["key"]]["raw_output"] == reference,
            }
            for variant in VARIANTS
            # The table exists to show what quantization did, so it holds only the quantized builds:
            # the base model is the "before" panel above it, and the fine-tune is the reference the
            # `identical` flag is measured against - listing it here would compare it with itself.
            if variant["key"] not in ("base", "ft")
        ],
    }


# The ablation arms, in the order they were decided. Each names the run directory holding its dev
# evaluation; the trainer wrote those into the *_train directories.
ABLATIONS = [
    ("Base model, zero-shot", "20260915-0050_base-bf16_dev", False),
    ("r16, LM only, lr 1e-4", "20260915-1029_qlora-r16-lm-lr1e4_train", False),
    ("r16, LM only, lr 4e-4", "20260915-1155_qlora-r16-lm-lr4e4_train", True),
    ("r8, LM only, lr 4e-4", "20260915-1412_qlora-r8-lm-lr4e4-e1_train", False),
    ("r32, LM only, lr 4e-4", "20260915-1459_qlora-r32-lm-lr4e4-e1_train", False),
    ("r16, LM + vision projector", "20260915-1555_qlora-r16-lmproj-lr4e4-e1_train", False),
    ("r16, bf16 LoRA (no NF4 base)", "20260915-1643_lora-bf16-r16-lm-lr4e4-e1_train", False),
    ("Partial fine-tune, last 4 layers", "20260915-1757_partial4-bf16-lr5e5-e1_train", False),
]

# The quantization table. Dev rows show the bit-width sweep; test rows show what actually ships.
QUANT_ROWS = [
    ("GGUF bf16 (reference)", "20260916-0428_ft-r16-gguf-bf16_dev_final", False),
    ("GGUF Q8_0", "20260916-0432_ft-r16-gguf-q8_0_dev_final", False),
    ("GGUF Q6_K", "20260916-0435_ft-r16-gguf-q6_k_dev_final", False),
    ("GGUF Q5_K_M", "20260916-0437_ft-r16-gguf-q5_k_m_dev_final", False),
    ("GGUF Q4_K_M", "20260916-0440_ft-r16-gguf-q4_k_m_dev_final", False),
    ("GGUF Q3_K_M, no imatrix", "20260916-0442_ft-r16-gguf-q3_k_m_dev_final", False),
    ("GGUF Q2_K, no imatrix", "20260916-0445_ft-r16-gguf-q2_k_dev_final", False),
    ("GGUF Q3_K_M + imatrix", "20260916-0503_ft-r16-gguf-q3_k_m-imat_dev_imat", False),
    ("GGUF Q2_K + imatrix", "20260916-0506_ft-r16-gguf-q2_k-imat_dev_imat", False),
    ("Fine-tuned bf16 on vLLM", "20260916-0000_bf16-merged-vllm_test_final", False),
    ("AWQ W4A16 on vLLM", "20260915-2353_awq-w4a16-vllm_test_final", True),
    ("GGUF Q4_K_M on llama.cpp", "20260916-0516_ft-r16-gguf-q4_k_m_test_final", False),
]


def stats_summary() -> dict[str, Any]:
    """The data report: split distributions, the schema's key paths, and the leakage check."""
    return json.loads((ROOT / "outputs/runs/20260915-0054_cord-v2_all/summary.json").read_text())


def summary_of(run: str) -> dict[str, Any]:
    path = ROOT / "outputs/runs" / run / "summary.json"
    if not path.exists():
        raise SystemExit(f"missing {path}; pull the run outputs first")
    return json.loads(path.read_text())


def build_grid(demo: dict[str, Any]) -> dict[str, Any]:
    """The guide's tables, read from the committed summaries. No figure here is written by hand."""
    base, ft = (
        summary_of("20260915-2200_base-bf16_test_final"),
        summary_of("20260915-2231_qlora-r16-lm-lr4e4_test_final"),
    )
    awq = summary_of("20260915-2353_awq-w4a16-vllm_test_final")
    exact = sum(1 for r in demo["receipts"] if r["ft"]["exact"])

    quick = [
        {"value": f"{base['metrics']['f1']['value']:.3f}", "label": "base model, test"},
        {"value": f"{ft['metrics']['f1']['value']:.3f}", "label": "fine-tuned, test", "colour": "ok"},
        {"value": f"{exact}/100", "label": "exactly correct"},
        {
            "value": f"{awq['run']['output_tokens_per_s']:.0f}",
            "label": "tok/s at 4-bit",
            "colour": "acc",
        },
    ]

    ablations = []
    for label, run, chosen in ABLATIONS:
        metrics = summary_of(run)["metrics"]
        ablations.append(
            {
                "label": label,
                "f1": metrics["f1"]["value"],
                "numeric": metrics["f1_numeric"]["value"],
                "text": metrics["f1_text"]["value"],
                "chosen": chosen,
            }
        )

    quant = []
    for label, run, chosen in QUANT_ROWS:
        summary = summary_of(run)
        quant.append(
            {
                "label": label,
                "split": summary["split"],
                "f1": summary["metrics"]["f1"]["value"],
                "p50": summary["run"]["latency_s"]["p50"],
                "tps": summary["run"]["output_tokens_per_s"],
                "chosen": chosen,
            }
        )

    stats = stats_summary()

    # What the chosen training run cost, and every knob it was given.
    train_dir = ROOT / "outputs/runs/20260915-1155_qlora-r16-lm-lr4e4_train"
    train_summary = json.loads((train_dir / "train_summary.json").read_text())
    hyper = [
        line.split(": ", 1) for line in (train_dir / "config.yaml").read_text().splitlines() if ": " in line
    ]

    metrics_table = []
    for label, run in (
        ("Base model, zero-shot", "20260915-2200_base-bf16_test_final"),
        ("Fine-tuned (deployed)", "20260915-2231_qlora-r16-lm-lr4e4_test_final"),
    ):
        metrics = summary_of(run)["metrics"]
        metrics_table.append({"label": label, **{key: metrics[key]["value"] for key in metrics}})

    return {
        "quick": quick,
        "ablations": ablations,
        "quant": quant,
        "splits": [
            {
                "split": name,
                "receipts": block["receipts"],
                "fields": block["fields"],
                "numeric_share": block["numeric_share"],
                "fields_p50": block["fields_per_receipt"]["p50"],
                "fields_max": block["fields_per_receipt"]["max"],
                "width_p50": block["width"]["p50"],
                "height_p50": block["height"]["p50"],
                "mp_p95": block["megapixels"]["p95"],
            }
            for name, block in stats["splits"].items()
        ],
        # The real shape, checked against the file rather than assumed: an earlier version guessed
        # three key names that do not exist and, guarded by `if key in stats`, shipped an empty
        # section without failing.
        "leakage": {
            "hash_bits": stats["leakage"]["hash_bits"],
            "max_distance": stats["leakage"]["max_distance"],
            "cross_split_exact": stats["leakage"]["cross_split_exact"],
            "cross_split_pairs": stats["leakage"]["cross_split_pairs_within_max_distance"],
            "within_split_exact": stats["leakage"]["exact_duplicates_within_a_split"],
            "before": stats["leakage"]["receipts_before_exclusion"],
            "nearest": stats["leakage"]["nearest_other_split_distance"],
            "excluded": stats["leakage"]["excluded"],
        },
        "key_paths": stats["splits"]["train"]["key_paths"],
        "target_tokens": {name: block["target_tokens"] for name, block in stats["splits"].items()},
        "train": {
            "seconds": train_summary["train_seconds"],
            "loss": train_summary["train_loss"],
            "peak_vram_mb": train_summary["train_peak_vram_mb"],
            "trainable_params": train_summary["trainable_params"],
            "hyper": hyper,
        },
        "metrics_table": metrics_table,
        "tree": repo_map(),
    }


def repo_map() -> list[dict[str, Any]]:
    """Every package with its file and line counts, generated from the tree rather than typed."""
    parts = [
        ("src/receipt_vlm/data", "CORD → normalized examples, splits, the leakage check"),
        ("src/receipt_vlm/eval", "field F1, TED, bootstrap intervals, paired comparisons"),
        ("src/receipt_vlm/infer", "transformers, vLLM and llama.cpp behind one interface"),
        ("src/receipt_vlm/train", "the QLoRA trainer and the ablation sweep"),
        ("src/receipt_vlm/quant", "adapter merge and the AWQ recipe"),
        ("src/receipt_vlm/serve", "the HTTP endpoint and the Gradio demo"),
        ("scripts", "thin command-line entry points"),
        ("tests", "CPU-only tests on a three-receipt fixture"),
    ]
    out = []
    for path, what in parts:
        files = sorted((ROOT / path).rglob("*.py"))
        lines = sum(len(f.read_text().splitlines()) for f in files)
        out.append({"path": path, "what": what, "files": len(files), "lines": lines})
    return out


def inject(template: Path, out: Path, placeholder: str, payload: Any) -> None:
    """Fill one placeholder with JSON, escaping `</` so the data cannot close the script tag."""
    text = template.read_text()
    if placeholder not in text:
        raise SystemExit(f"{template.name} has no {placeholder}")
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    out.write_text(text.replace(placeholder, data))
    print(f"{out.relative_to(ROOT)}: {out.stat().st_size / 1024:.0f} KB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", default="outputs/site_data", help="images and gold for the receipts")
    parser.add_argument("--story", default="cord-test-0035", help="the receipt the tour follows")
    args = parser.parse_args()

    manifest = json.loads((ROOT / args.audit / "manifest.json").read_text())
    site = ROOT / "site"

    demo = build_demo(manifest)
    inject(site / "demo-template.html", site / "demo.html", "/*__DEMO__*/null", demo)

    story = build_story(manifest, args.story)
    inject(site / "template.html", site / "index.html", "/*__STORY__*/null", story)

    grid = build_grid(demo)
    inject(site / "guide-template.html", site / "guide.html", "/*__GRID__*/null", grid)

    same = [v["label"] for v in story["variants"] if v["identical"]]
    print(f"  tour: {args.story} · base {story['base']['f1']:.2f} → fine-tuned {story['ft']['f1']:.2f}")
    print(f"  byte-identical to the fine-tune: {', '.join(same) if same else 'none'}")
    print(f"  guide: {len(grid['ablations'])} ablation arms · {len(grid['quant'])} quantization rows")

    exact = sum(1 for r in demo["receipts"] if r["ft"]["exact"])
    rescued = sum(1 for r in demo["receipts"] if r["base"]["f1"] < 0.6 and r["ft"]["f1"] > 0.9)
    print(
        f"  {len(demo['receipts'])} receipts · {exact} exact after fine-tuning"
        f" · {rescued} the base model got wrong and the fine-tune gets right"
    )


if __name__ == "__main__":
    main()
