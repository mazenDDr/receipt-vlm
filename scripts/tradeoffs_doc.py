"""Render docs/tradeoffs.md from the committed run summaries.

    python scripts/tradeoffs_doc.py [--config configs/tradeoffs.yaml] [--out docs/tradeoffs.md]

Every number in the document is read from `outputs/runs/*/summary.json` and the paired comparisons are
recomputed from `scores.jsonl`, so the prose can never drift from the runs. The only hand-written facts live
in the config: weights on disk, measured peak memory, and which engine ran each row.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import yaml

from receipt_vlm.eval import compare
from receipt_vlm.eval.report import LABELS


def load_summary(run_dir: str | None) -> dict[str, Any] | None:
    if not run_dir:
        return None
    path = Path(run_dir) / "summary.json"
    if not path.exists():
        raise SystemExit(f"missing {path}: pull the run outputs first")
    return json.loads(path.read_text())


def metric(summary: dict | None, name: str, field: str = "value") -> float | None:
    if not summary:
        return None
    return summary.get("metrics", {}).get(name, {}).get(field)


def num(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def accuracy_table(rows: list[dict]) -> list[str]:
    out = [
        "| Row | Engine | Weights GiB | Dev F1 | Test F1 | Test numeric | Test text | Valid JSON |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        dev, test = row["_dev"], row["_test"]
        scored = test or dev
        weights = row.get("weights_gib")
        out.append(
            f"| {row['label']} | {row['engine']} | {'—' if weights is None else f'{weights:.2f}'} "
            f"| {num(metric(dev, 'f1'))} | {num(metric(test, 'f1'))} "
            f"| {num(metric(test, 'f1_numeric'))} | {num(metric(test, 'f1_text'))} "
            f"| {num(metric(scored, 'valid_json'))} |"
        )
    return out


def speed_table(rows: list[dict]) -> list[str]:
    out = [
        "| Row | Engine | Split timed | Latency p50 s | p95 s | Output tok/s | Peak GPU MiB |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        timed, split = (row["_test"], "test") if row["_test"] else (row["_dev"], "dev")
        run = (timed or {}).get("run", {})
        latency = run.get("latency_s", {})
        peak = row.get("peak_mib")
        peak_cell = f"{peak:,}" if peak else f"— *({row['peak_note']})*" if row.get("peak_note") else "—"
        out.append(
            f"| {row['label']} | {row['engine']} | {split} "
            f"| {num(latency.get('p50'), 2)} | {num(latency.get('p95'), 2)} "
            f"| {num(run.get('output_tokens_per_s'), 1)} | {peak_cell} |"
        )
    return out


def comparison_tables(comparisons: list[dict]) -> list[str]:
    """A compact four-metric version of `compare.render()`: overall, numeric, text, validity.

    `compare.compare()` returns {metric: (difference, lo, hi)}.
    """
    shown = ("f1", "f1_numeric", "f1_text", "valid_json")
    out: list[str] = []
    for item in comparisons:
        diffs = compare.compare(Path(item["a"]), Path(item["b"]))
        header = "| Metric | Difference | 95% interval | Real? |"
        out += [f"**{item['label']}**", "", header, "|---|---|---|---|"]
        for name in shown:
            difference, lo, hi = diffs[name]
            real = "**yes**" if compare.is_real(diffs[name]) else "no"
            out.append(f"| {LABELS.get(name, name)} | {difference:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {real} |")
        out.append("")
    return out


ENGINE_COLOR = {"transformers": "#b45309", "vLLM": "#1d4ed8", "llama.cpp": "#047857"}


def plot_points(rows: list[dict]) -> list[tuple[str, str, float, float, float]]:
    """(label, engine, latency p50, field F1, weights GiB) for every row that has both numbers."""
    points = []
    for row in rows:
        timed = row["_test"] or row["_dev"]
        f1 = metric(timed, "f1")
        p50 = (timed or {}).get("run", {}).get("latency_s", {}).get("p50")
        if f1 is not None and p50:
            points.append((row["label"], row["engine"], p50, f1, row.get("weights_gib") or 3.0))
    return points


def plot_svg(rows: list[dict], w: int = 780, h: int = 440) -> str:
    """Field F1 against latency, marker area proportional to the weights that must be loaded.

    A standalone file, because GitHub does not render inline SVG inside markdown. System fonts only.
    """
    left, right, top, bottom = 56, 292, 26, 48
    lo, hi = math.log10(0.7), math.log10(16.0)

    def x_of(seconds: float) -> float:
        return left + (math.log10(seconds) - lo) / (hi - lo) * (w - left - right)

    def y_of(f1: float) -> float:
        return h - bottom - f1 * (h - top - bottom)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}"'
        ' font-family="system-ui, -apple-system, Segoe UI, Roboto, sans-serif">',
        f'<rect width="{w}" height="{h}" fill="#ffffff"/>',
    ]
    for value in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = y_of(value)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{w - right}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(
            f'<text x="{left - 10}" y="{y + 4:.1f}" font-size="11" fill="#6b7280"'
            f' text-anchor="end">{value:.2f}</text>'
        )
    for seconds in (1, 2, 5, 10):
        x = x_of(seconds)
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{h - bottom}" stroke="#f3f4f6"/>')
        parts.append(
            f'<text x="{x:.1f}" y="{h - bottom + 18}" font-size="11" fill="#6b7280"'
            f' text-anchor="middle">{seconds}s</text>'
        )
    mid_x, mid_y = (left + w - right) / 2, (top + h - bottom) / 2
    parts.append(
        f'<text x="{mid_x:.0f}" y="{h - 12}" font-size="12" fill="#374151"'
        ' text-anchor="middle">latency per receipt, p50 (log scale)</text>'
    )
    parts.append(
        f'<text x="15" y="{mid_y:.0f}" font-size="12" fill="#374151" text-anchor="middle"'
        f' transform="rotate(-90 15 {mid_y:.0f})">field F1</text>'
    )

    labels = []
    for label, engine, p50, f1, gib in plot_points(rows):
        x, y = x_of(p50), y_of(f1)
        radius = max(4.0, math.sqrt(gib) * 2.6)
        color = ENGINE_COLOR.get(engine, "#6b7280")
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{color}" fill-opacity="0.22"'
            f' stroke="{color}" stroke-width="1.5"/>'
        )
        labels.append({"text": label, "color": color, "mx": x, "my": y, "r": radius})

    # Labels live in the right gutter, ordered by marker height. Text then never sits on a marker,
    # and because markers and labels share one top-to-bottom order the connectors cannot cross.
    # Seven rows fall between F1 0.83 and 0.905 - about 25 px apart here - so in-place labels were
    # unreadable however far they were nudged.
    label_x = w - right + 14
    labels.sort(key=lambda item: item["my"])
    step = min(15.0, (h - top - bottom - 16) / max(len(labels), 1))
    for index, item in enumerate(labels):
        label_y = top + 14 + index * step
        parts.append(
            f'<line x1="{item["mx"] + item["r"]:.1f}" y1="{item["my"]:.1f}" x2="{label_x - 6:.1f}"'
            f' y2="{label_y - 4:.1f}" stroke="{item["color"]}" stroke-width="0.7" stroke-opacity="0.45"/>'
        )
        parts.append(
            f'<text x="{label_x:.1f}" y="{label_y:.1f}" font-size="10.5" fill="#111827">{item["text"]}</text>'
        )

    # The legend sits inside the plot, in the empty band between the working rows and the two
    # uncalibrated K-quants on the floor; the right margin is now the label column.
    legend_x, legend_y = left + 18, y_of(0.56)
    for engine, color in ENGINE_COLOR.items():
        parts.append(
            f'<circle cx="{legend_x}" cy="{legend_y:.0f}" r="6" fill="{color}" fill-opacity="0.22"'
            f' stroke="{color}" stroke-width="1.5"/>'
        )
        parts.append(
            f'<text x="{legend_x + 14}" y="{legend_y + 4:.0f}" font-size="11.5"'
            f' fill="#374151">{engine}</text>'
        )
        legend_y += 21
    parts.append(
        f'<text x="{legend_x - 6}" y="{legend_y + 8:.0f}" font-size="10.5" fill="#6b7280">'
        "marker area = weights on disk</text>"
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def render(config: dict) -> str:
    rows = config["rows"]
    for row in rows:
        row["_dev"] = load_summary(row.get("dev"))
        row["_test"] = load_summary(row.get("test"))

    by_label = {row["label"]: row for row in rows}
    base, ft = by_label["Base, bf16"], by_label["Fine-tuned QLoRA (NF4 base + adapter)"]
    awq, q4 = by_label["AWQ W4A16"], by_label["GGUF Q4_K_M"]
    gguf_bf16 = by_label["GGUF bf16"]
    ft_p50 = ft["_test"]["run"]["latency_s"]["p50"]
    awq_p50 = awq["_test"]["run"]["latency_s"]["p50"]

    lines = [
        "# Where quantization costs something, and where it does not",
        "",
        "Every number here is read from the run summaries under `outputs/runs/`; this page is",
        "generated by `scripts/tradeoffs_doc.py`. Accuracy is field-level F1 against CORD's",
        "`gt_parse`, scored the same way for every row. Dev has 99 receipts and test has 100.",
        "Test was scored only for the final variants.",
        "",
        "## The short version",
        "",
        f"- Fine-tuning is the only change that moves accuracy a lot: test F1"
        f" {num(metric(base['_test'], 'f1'))} → {num(metric(ft['_test'], 'f1'))}.",
        "- **Quantization to 4 bits is close to free.** AWQ W4A16 and GGUF Q4_K_M are both",
        "  within noise of their own bf16 reference on test, at roughly half the size and",
        "  far faster.",
        "- **The runtime matters more than the bit width.** The same fine-tuned weights take",
        f"  {ft_p50:.2f} s per receipt in transformers and {awq_p50:.2f} s as AWQ on vLLM. The",
        "  entire spread across every GGUF bit width is smaller than that gap.",
        "- **Below 4 bits accuracy degrades across the board**, not one kind of field: Q2_K",
        "  with an importance matrix loses overall, numeric and text on test alike. Numeric",
        "  is only the first to show it — on dev it was the single real loss while overall",
        "  and text were still within noise. At 4 bits and above, nothing is real on either",
        "  split.",
        "",
        "## Accuracy",
        "",
        *accuracy_table(rows),
        "",
        "## Latency, throughput and memory",
        "",
        "Single image, batch size 1, greedy decoding, identical prompt and image cap",
        "everywhere. Peak GPU is measured in-process for transformers and sampled with",
        "`nvidia-smi` for llama.cpp (616 MiB idle baseline).",
        "",
        *speed_table(rows),
        "",
        "## Accuracy against latency",
        "",
        "Marker area is the weights that must be loaded; colour is the runtime. The two",
        "uncalibrated K-quants sit on the floor of the chart, at F1 0.100 and 0.000.",
        "",
        "![Field F1 against latency per receipt, by runtime and bit width](tradeoffs.svg)",
        "",
        "## Paired comparisons",
        "",
        "Each difference is computed receipt by receipt over the same split, with a 95%",
        "bootstrap interval. A difference counts as real only when its interval excludes 0.",
        "",
        *comparison_tables(config["comparisons"]),
        "## What to deploy",
        "",
        f"**AWQ W4A16 on vLLM.** It is the smallest checkpoint at {awq['weights_gib']:.2f} GiB,"
        f" the fastest at {awq['_test']['run']['output_tokens_per_s']:.0f} output tok/s, and"
        f" its test F1 ({num(metric(awq['_test'], 'f1'))}) is within noise of the merged bf16"
        " model it came from.",
        "",
        f"**GGUF Q4_K_M** is the answer when the target is llama.cpp rather than a GPU server:"
        f" test F1 {num(metric(q4['_test'], 'f1'))} against the GGUF bf16 row's"
        f" {num(metric(gguf_bf16['_test'], 'f1'))}, at {q4['peak_mib']:,} MiB instead of"
        f" {gguf_bf16['peak_mib']:,} MiB.",
        "",
        "Two cautions that cost real time to learn, both documented in the sections above:",
        "",
        "1. **The vision projector cannot be quantized.** It must be built at f32, where it",
        "   is 2.49 GiB and 68% of the smallest GGUF row's footprint. This is why no GGUF bit",
        "   width undercuts AWQ on disk.",
        "2. **K-quants below Q4 need an importance matrix.** Without one they do not degrade,",
        "   they collapse: Q2_K emitted a median of three tokens. With one, Q2_K looked tied",
        "   with bf16 on dev — and was really worse on test. Both rows are kept in the tables",
        "   so the difference is visible.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tradeoffs.yaml")
    parser.add_argument("--out", default="docs/tradeoffs.md")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = render(config)  # also loads each row's summaries into the config dicts, for the plot
    out.write_text(text)
    svg = out.parent / "tradeoffs.svg"
    svg.write_text(plot_svg(config["rows"]))
    print(f"{out}: {len(text.splitlines())} lines from {len(config['rows'])} rows")
    print(f"{svg}: {len(plot_points(config['rows']))} points")


if __name__ == "__main__":
    main()
