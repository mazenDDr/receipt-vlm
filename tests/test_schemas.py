from receipt_vlm.schemas import ExampleScore, FieldCounts, Prediction, ReceiptExample


def test_example_round_trips_through_json():
    example = ReceiptExample(
        example_id="cord-test-0000",
        split="test",
        image_path="data/processed/images/cord-test-0000.png",
        image_sha256="0" * 64,
        width=432,
        height=648,
        target={
            "menu": {"nm": "-TICKET CP", "cnt": "2", "price": "60.000"},
            "total": {"total_price": "60.000"},
        },
        n_fields=4,
    )
    assert ReceiptExample.model_validate_json(example.model_dump_json()) == example


def test_prediction_allows_unparsed_output():
    pred = Prediction(
        example_id="cord-test-0000",
        variant="base-bf16",
        raw_output='{"menu": ',
        parsed=None,
        prompt_tokens=900,
        output_tokens=512,
        latency_s=3.2,
        truncated=True,
    )
    assert pred.parsed is None and pred.peak_vram_mb is None


def test_score_counts_default_to_zero():
    score = ExampleScore(
        example_id="cord-test-0000",
        variant="base-bf16",
        valid_json=False,
        exact_match=False,
        overall=FieldCounts(fn=4),
        numeric=FieldCounts(fn=3),
        text=FieldCounts(fn=1),
    )
    assert score.overall.tp == 0 and score.overall.fn == 4
