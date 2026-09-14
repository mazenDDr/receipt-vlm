"""Build the processed CORD v2 dataset on the GPU machine and write a data report.

    python scripts/build_data.py --config configs/data.yaml --tokenizer Qwen/Qwen2.5-VL-3B-Instruct

Writes data/processed/{examples.jsonl, images/} and outputs/runs/<run_id>/ (config, summary.json, report.md).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from PIL import Image

from receipt_vlm import config
from receipt_vlm.data import build, dedup, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--tokenizer", default=None, help="HF model id; adds target token counts")
    args = parser.parse_args()
    cfg = config.load(args.config, build.DataConfig)
    run_dir = Path("outputs/runs") / f"{time.strftime('%Y%m%d-%H%M')}_cord-v2_all"
    config.save(cfg, run_dir / "config.yaml")

    t0 = time.perf_counter()
    raw, out = Path(cfg.raw_dir), Path(cfg.out_dir)
    examples = build.build(((s, build.read_rows(raw, s)) for s in build.SPLITS), out)
    print(f"built {len(examples)} examples in {time.perf_counter() - t0:.1f}s")

    hashes = {}
    for e in examples:
        with Image.open(e.image_path) as image:
            hashes[e.example_id] = dedup.dhash(image, cfg.hash_size)
    pairs = dedup.nearest_pairs(examples, hashes, cfg.near_dup_max_distance)
    margins = dedup.nearest_distance(examples, hashes)
    exact_within = len(examples) - len({(e.split, e.image_sha256) for e in examples})

    count_tokens = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        count_tokens = lambda text: len(tokenizer(text)["input_ids"])  # noqa: E731
    per_split = stats.split_stats(examples, count_tokens)

    summary = {
        "splits": per_split,
        "leakage": {
            "hash_bits": cfg.hash_size**2,
            "cross_split_exact": sum(p.exact for p in pairs),
            "cross_split_pairs_within_max_distance": len(pairs),
            "max_distance": cfg.near_dup_max_distance,
            "closest_pairs": [p.model_dump() for p in pairs[:20]],
            "nearest_other_split_distance": stats.percentiles([float(d) for d in margins.values()]),
            "exact_duplicates_within_a_split": exact_within,
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (run_dir / "report.md").write_text(render(summary))
    print(render(summary))
    print(f"wrote {run_dir}")


def render(summary: dict) -> str:
    lines = [
        "# CORD v2 data report",
        "",
        "| Split | Receipts | Fields | Numeric share | Fields/receipt p50 (max) |",
    ]
    lines.append("|---|---|---|---|---|")
    for split, s in summary["splits"].items():
        fpr = s["fields_per_receipt"]
        cells = [
            split,
            s["receipts"],
            s["fields"],
            f"{s['numeric_share']:.0%}",
            f"{fpr['p50']:.0f} ({fpr['max']:.0f})",
        ]
        lines.append("| " + " | ".join(map(str, cells)) + " |")
    if any("target_tokens" in s for s in summary["splits"].values()):
        lines += ["", "| Split | Target tokens p50 | p95 | max |", "|---|---|---|---|"]
        for split, s in summary["splits"].items():
            t = s["target_tokens"]
            lines.append(f"| {split} | {t['p50']:.0f} | {t['p95']:.0f} | {t['max']:.0f} |")
    leak = summary["leakage"]
    lines += [
        "",
        "## Leakage check",
        f"- Identical images across splits: **{leak['cross_split_exact']}**",
        f"- Cross-split pairs within {leak['max_distance']} of {leak['hash_bits']} hash bits: "
        f"**{leak['cross_split_pairs_within_max_distance']}**",
        f"- Distance from each dev/test receipt to its nearest receipt in another split: "
        f"{leak['nearest_other_split_distance']}",
        f"- Identical images within a split: {leak['exact_duplicates_within_a_split']}",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
