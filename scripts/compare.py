"""Compare two scored runs receipt by receipt (paired bootstrap).

    python scripts/compare.py outputs/runs/<fine-tuned run> outputs/runs/<baseline run>

Prints each metric's difference (first minus second) with a 95% interval. Both runs must cover the same
receipts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from receipt_vlm.eval import compare
from receipt_vlm.eval.report import load_scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    args = parser.parse_args()
    diffs = compare.compare(args.a, args.b)
    n = len(load_scores(args.a / "scores.jsonl"))
    print(compare.render(args.a.name, args.b.name, diffs, n))


if __name__ == "__main__":
    main()
