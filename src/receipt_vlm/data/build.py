"""CORD v2 parquet rows → ReceiptExample JSONL, plus image files with their original bytes."""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from PIL import Image
from pydantic import BaseModel

from receipt_vlm.eval.fields import flatten
from receipt_vlm.schemas import ReceiptExample, Split

# CORD's split name → ours. CORD validation is where we tune and select; test is scored once at the end.
SPLITS: dict[str, Split] = {"train": "train", "validation": "dev", "test": "test"}
_EXTENSIONS = {"JPEG": "jpg", "PNG": "png"}


class DataConfig(BaseModel):
    # filled by `hf download naver-clova-ix/cord-v2 --repo-type dataset --local-dir <raw_dir>`
    raw_dir: str = "data/raw/cord-v2"
    out_dir: str = "data/processed"
    hash_size: int = 16  # difference hash of hash_size x hash_size bits
    near_dup_max_distance: int = 20  # report cross-split pairs at or below this Hamming distance


def read_rows(raw_dir: Path, cord_split: str) -> Iterator[dict[str, Any]]:
    """Rows in the order the HF `datasets` loader gives them: files sorted by name, then row order."""
    import pyarrow.parquet as pq  # only on the GPU machine

    for path in sorted((raw_dir / "data").glob(f"{cord_split}-*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32, columns=["image", "ground_truth"]):
            yield from batch.to_pylist()


def example_from_row(row: dict[str, Any], cord_split: str, index: int, out_dir: Path) -> ReceiptExample:
    image_bytes: bytes = row["image"]["bytes"]
    target = json.loads(row["ground_truth"])["gt_parse"]
    example_id = f"cord-{cord_split}-{index:04d}"
    with Image.open(io.BytesIO(image_bytes)) as image:
        ext = _EXTENSIONS.get(image.format or "", (image.format or "bin").lower())
        width, height = image.size
    image_path = out_dir / "images" / f"{example_id}.{ext}"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(image_bytes)
    return ReceiptExample(
        example_id=example_id,
        split=SPLITS[cord_split],
        image_path=image_path.as_posix(),
        image_sha256=hashlib.sha256(image_bytes).hexdigest(),
        width=width,
        height=height,
        target=target,
        n_fields=len(flatten(target)),
    )


def build(
    rows_by_split: Iterable[tuple[str, Iterable[dict[str, Any]]]], out_dir: Path
) -> list[ReceiptExample]:
    """Write every example's image and `out_dir/examples_all.jsonl` (before the leakage exclusions)."""
    examples = [
        example_from_row(row, cord_split, index, out_dir)
        for cord_split, rows in rows_by_split
        for index, row in enumerate(rows)
    ]
    write_examples(examples, out_dir / "examples_all.jsonl")
    return examples


def write_examples(examples: Iterable[ReceiptExample], path: Path) -> None:
    path.write_text("".join(e.model_dump_json() + "\n" for e in examples))


def load_examples(path: str | Path) -> list[ReceiptExample]:
    return [ReceiptExample.model_validate_json(line) for line in Path(path).read_text().splitlines() if line]
