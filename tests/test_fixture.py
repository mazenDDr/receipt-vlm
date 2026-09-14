"""Checks on three real CORD v2 receipts (one per split) in tests/fixtures/cord_mini/."""

import hashlib
from pathlib import Path

from PIL import Image

from receipt_vlm.data.build import load_examples
from receipt_vlm.eval import metrics
from receipt_vlm.eval.parse import extract_json
from receipt_vlm.prompts import serialize_target
from receipt_vlm.schemas import Prediction

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = load_examples(ROOT / "tests/fixtures/cord_mini/examples.jsonl")


def test_fixture_has_one_receipt_per_split_with_matching_image_bytes():
    assert sorted(e.split for e in EXAMPLES) == ["dev", "test", "train"]
    for e in EXAMPLES:
        data = (ROOT / e.image_path).read_bytes()
        assert hashlib.sha256(data).hexdigest() == e.image_sha256
        with Image.open(ROOT / e.image_path) as image:
            assert image.size == (e.width, e.height)


def test_real_targets_survive_the_training_round_trip_and_score_perfectly():
    for e in EXAMPLES:
        parsed = extract_json(serialize_target(e.target))
        assert parsed == e.target
        pred = Prediction(
            example_id=e.example_id,
            variant="gold",
            raw_output=serialize_target(e.target),
            parsed=parsed,
            prompt_tokens=0,
            output_tokens=0,
            latency_s=0.0,
        )
        s = metrics.score(e, pred)
        assert s.exact_match and s.overall.tp == e.n_fields and s.ted_accuracy == 1.0
