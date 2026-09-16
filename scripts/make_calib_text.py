"""Plain-text calibration corpus for llama.cpp's importance matrix (llama-imatrix).

    python scripts/make_calib_text.py --split train --out data/processed/calib.txt

llama-imatrix reads text, not images, and that is the right scope: the matrix weights the language model,
which is the only part being quantized. The vision projector stays f32 (at f16 it overflows into NaN), so
nothing in the vision path is covered by it.

Each receipt contributes the instruction the model is actually served with, followed by the JSON it is
trained to emit, so the matrix is measured on the token distribution this model really produces rather than
on generic web text.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from receipt_vlm.data.build import load_examples
from receipt_vlm.prompts import INSTRUCTION, serialize_target


def build_text(examples: list, instruction: str = INSTRUCTION) -> str:
    """One block per receipt: the instruction, then the target JSON."""
    blocks = [f"{instruction}\n{serialize_target(e.target)}" for e in examples]
    return "\n\n".join(blocks) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", help="calibrate on train only; never dev or test")
    parser.add_argument("--examples-path", default="data/processed/examples.jsonl")
    parser.add_argument("--out", default="data/processed/calib.txt")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    if args.split != "train":
        raise SystemExit(f"calibration must use train, not {args.split!r}: dev and test stay unseen")
    examples = [e for e in load_examples(args.examples_path) if e.split == args.split][: args.limit]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_text(examples))
    print(f"{out}: {len(examples)} receipts, {out.stat().st_size} bytes")


if __name__ == "__main__":
    main()
