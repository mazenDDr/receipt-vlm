import io
import json

from PIL import Image, ImageDraw

from receipt_vlm.data import build, dedup, stats
from receipt_vlm.prompts import serialize_target

TARGET = {"menu": [{"nm": "EGG TART", "price": "13,000"}], "total": {"total_price": "13,000"}}


def _png(seed: int, size=(60, 90)) -> bytes:
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    for i in range(6):
        y = (seed * 7 + i * 13) % size[1]
        draw.rectangle([5, y, 5 + (seed * 11 + i * 17) % 50, y + 4], fill="black")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _row(seed: int, target=TARGET) -> dict:
    return {"image": {"bytes": _png(seed)}, "ground_truth": json.dumps({"gt_parse": target, "meta": {}})}


def test_build_writes_original_bytes_and_stable_ids(tmp_path):
    rows = [("train", [_row(1), _row(2)]), ("validation", [_row(3)]), ("test", [_row(4)])]
    examples = build.build(rows, tmp_path)
    assert [e.example_id for e in examples] == [
        "cord-train-0000",
        "cord-train-0001",
        "cord-validation-0000",
        "cord-test-0000",
    ]
    assert [e.split for e in examples] == ["train", "train", "dev", "test"]
    assert (tmp_path / "images" / "cord-test-0000.png").read_bytes() == _png(4)
    assert examples[0].target == TARGET and examples[0].n_fields == 3
    assert (examples[0].width, examples[0].height) == (60, 90)
    assert build.load_examples(tmp_path / "examples.jsonl") == examples


def test_dhash_finds_copies_across_splits_but_not_within(tmp_path):
    rows = [("train", [_row(1), _row(2)]), ("validation", [_row(1)]), ("test", [_row(9)])]
    examples = build.build(rows, tmp_path)
    hashes = {e.example_id: dedup.dhash(Image.open(e.image_path)) for e in examples}
    pairs = dedup.nearest_pairs(examples, hashes, max_distance=0)
    assert [(p.a, p.b, p.exact) for p in pairs] == [("cord-train-0000", "cord-validation-0000", True)]
    margins = dedup.nearest_distance(examples, hashes)
    assert margins["cord-validation-0000"] == 0 and margins["cord-test-0000"] > 0


def test_dhash_survives_resizing():
    image = Image.open(io.BytesIO(_png(3)))
    assert dedup.hamming(dedup.dhash(image), dedup.dhash(image.resize((120, 180)))) <= 10


def test_split_stats_counts_kinds_and_tokens(tmp_path):
    examples = build.build([("train", [_row(1)]), ("test", [_row(2)])], tmp_path)
    out = stats.split_stats(examples, count_tokens=lambda text: len(text))
    assert out["train"]["receipts"] == 1 and out["train"]["fields"] == 3
    assert out["train"]["numeric_share"] == 2 / 3
    assert out["test"]["target_tokens"]["max"] == len(serialize_target(TARGET))
    assert "dev" not in out


def test_serialize_target_is_compact_and_keeps_order_and_unicode():
    assert serialize_target({"total": {"total_price": "5"}, "menu": {"nm": "Café"}}) == (
        '{"total":{"total_price":"5"},"menu":{"nm":"Café"}}'
    )
