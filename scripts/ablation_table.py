"""A markdown table of scored runs on one split, e.g. every sweep arm next to the zero-shot baseline.

python scripts/ablation_table.py outputs/runs/<run> [outputs/runs/<run> ...]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from receipt_vlm.eval import tables


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    print(tables.render([tables.run_row(run) for run in args.runs]))


if __name__ == "__main__":
    main()
