"""Score a run directory's predictions and write scores.jsonl, summary.json and report.md next to them."""

from __future__ import annotations

import json
from pathlib import Path

from receipt_vlm.data.stats import percentiles
from receipt_vlm.eval import metrics
from receipt_vlm.schemas import ExampleScore, Prediction, ReceiptExample

LABELS = {
    "f1": "Field F1",
    "f1_numeric": "Field F1, numeric",
    "f1_text": "Field F1, text",
    "f1_numeric_lenient": "Field F1, numeric (lenient)",
    "precision": "Precision",
    "recall": "Recall",
    "valid_json": "Valid JSON",
    "exact_match": "Receipt exact match",
    "ted_accuracy": "TED accuracy",
}


def load_predictions(path: Path) -> dict[str, Prediction]:
    if not path.exists():
        return {}
    preds = [Prediction.model_validate_json(line) for line in path.read_text().splitlines() if line]
    return {p.example_id: p for p in preds}


def run_stats(preds: list[Prediction]) -> dict[str, object]:
    latency = sum(p.latency_s for p in preds)
    return {
        "truncated": sum(p.truncated for p in preds) / len(preds),
        "output_tokens": percentiles([p.output_tokens for p in preds]),
        "prompt_tokens": percentiles([p.prompt_tokens for p in preds]),
        "latency_s": percentiles([p.latency_s for p in preds]),
        "output_tokens_per_s": sum(p.output_tokens for p in preds) / latency if latency else 0.0,
        "peak_vram_mb": max((p.peak_vram_mb or 0.0) for p in preds),
    }


def write(run_dir: Path, examples: list[ReceiptExample], variant: str, split: str) -> dict[str, object]:
    preds = load_predictions(run_dir / "predictions.jsonl")
    missing = [e.example_id for e in examples if e.example_id not in preds]
    if missing:
        raise ValueError(f"{len(missing)} receipts have no prediction yet, e.g. {missing[:3]}")
    scores = [metrics.score(e, preds[e.example_id]) for e in examples]
    (run_dir / "scores.jsonl").write_text("".join(s.model_dump_json() + "\n" for s in scores))
    summary = {
        "variant": variant,
        "split": split,
        "n": len(scores),
        "metrics": metrics.summarize(scores),
        "run": run_stats([preds[e.example_id] for e in examples]),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (run_dir / "report.md").write_text(render(summary))
    return summary


def render(summary: dict) -> str:
    lines = [
        f"# {summary['variant']} on {summary['split']} ({summary['n']} receipts)",
        "",
        "| Metric | Value | 95% interval |",
        "|---|---|---|",
    ]
    for name, m in summary["metrics"].items():
        lines.append(f"| {LABELS.get(name, name)} | {m['value']:.3f} | [{m['lo']:.3f}, {m['hi']:.3f}] |")
    run = summary["run"]
    out, lat = run["output_tokens"], run["latency_s"]
    lines += [
        "",
        "| Run | Value |",
        "|---|---|",
        f"| Truncated at max_new_tokens | {run['truncated']:.1%} |",
        f"| Output tokens p50 / p95 / max | {out['p50']:.0f} / {out['p95']:.0f} / {out['max']:.0f} |",
        f"| Prompt tokens p50 | {run['prompt_tokens']['p50']:.0f} |",
        f"| Latency p50 / p95 (s) | {lat['p50']:.2f} / {lat['p95']:.2f} |",
        f"| Output tokens per second of wall time | {run['output_tokens_per_s']:.1f} |",
        f"| Peak VRAM (MB) | {run['peak_vram_mb']:.0f} |",
    ]
    return "\n".join(lines) + "\n"


def load_scores(path: Path) -> list[ExampleScore]:
    return [ExampleScore.model_validate_json(line) for line in path.read_text().splitlines() if line]
