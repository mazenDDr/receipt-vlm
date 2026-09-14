import pytest

from receipt_vlm.eval import metrics
from receipt_vlm.eval.bootstrap import interval, paired_difference
from receipt_vlm.eval.ted import ted_accuracy
from receipt_vlm.schemas import FieldCounts, Prediction, ReceiptExample

GOLD = {
    "menu": [
        {"nm": "EGG TART", "cnt": "1", "price": "13,000"},
        {"nm": "PIZZA TOAST", "cnt": "1", "price": "16,000"},
    ],
    "total": {"total_price": "29,000"},
}


def _example(target=GOLD, example_id="cord-test-0000"):
    return ReceiptExample(
        example_id=example_id,
        split="test",
        image_path=f"data/processed/images/{example_id}.png",
        image_sha256="0" * 64,
        width=100,
        height=100,
        target=target,
        n_fields=7,
    )


def _pred(parsed, example_id="cord-test-0000"):
    return Prediction(
        example_id=example_id,
        variant="base-bf16",
        raw_output="",
        parsed=parsed,
        prompt_tokens=0,
        output_tokens=0,
        latency_s=0.0,
    )


def test_item_order_costs_ted_but_not_field_f1():
    reordered = {"total": GOLD["total"], "menu": list(reversed(GOLD["menu"]))}
    s = metrics.score(_example(), _pred(reordered))
    assert s.exact_match and s.valid_json
    assert s.overall == FieldCounts(tp=7) and s.numeric == FieldCounts(tp=5) and s.text == FieldCounts(tp=2)
    assert s.ted_accuracy == pytest.approx(35 / 51)  # Donut's JSONParseEvaluator gives the same value


def test_misread_missing_and_extra_fields_are_counted_per_kind():
    pred = {
        "menu": [
            {"nm": "PIZZA TOAST", "cnt": "1", "price": "16.000"},  # separator differs from the gold "16,000"
            {"nm": "EGG TART", "cnt": "1", "price": "13,000"},
        ],
        "total": {"cashprice": "50,000"},  # total_price missing, cashprice not on the gold receipt
    }
    s = metrics.score(_example(), _pred(pred))
    assert not s.exact_match
    assert s.overall == FieldCounts(tp=5, fp=2, fn=2)
    assert s.numeric == FieldCounts(tp=3, fp=2, fn=2)
    assert s.text == FieldCounts(tp=2)
    assert s.numeric_lenient == FieldCounts(tp=4, fp=1, fn=1)  # "16.000" matches once formatting is forgiven


def test_duplicate_items_are_a_multiset():
    gold = {"menu": [{"nm": "TEA"}, {"nm": "TEA"}]}
    s = metrics.score(_example(gold), _pred({"menu": {"nm": "TEA"}}))
    assert s.overall == FieldCounts(tp=1, fn=1)


def test_invalid_json_scores_every_gold_field_as_missed():
    s = metrics.score(_example(), _pred(None))
    assert not s.valid_json and not s.exact_match
    assert s.overall == FieldCounts(fn=7)
    assert s.ted_accuracy == 0.0


def test_score_refuses_mismatched_receipts():
    with pytest.raises(ValueError):
        metrics.score(_example(), _pred(GOLD, example_id="cord-test-0001"))


def test_ted_accuracy_charges_character_edits_on_leaves():
    gold = {"total": {"total_price": "60.000"}}
    # empty -> gold inserts total, <subtree>, total_price (1 each) and a 6-character leaf = 9
    assert ted_accuracy({"total": {"total_price": "60.001"}}, gold) == pytest.approx(1 - 1 / 9)


def test_f1_and_rates_from_summed_counts():
    assert metrics.f1(FieldCounts(tp=5, fp=2, fn=2)) == pytest.approx(10 / 14)
    assert metrics.f1(FieldCounts()) == 1.0
    assert metrics.total([FieldCounts(tp=1, fn=1), None, FieldCounts(fp=2)]) == FieldCounts(tp=1, fp=2, fn=1)


def test_summarize_reports_every_metric_inside_its_interval():
    examples = [_example(example_id=f"cord-test-{i:04d}") for i in range(20)]
    scores = [
        metrics.score(e, _pred(GOLD if i % 2 else None, example_id=e.example_id))
        for i, e in enumerate(examples)
    ]
    summary = metrics.summarize(scores, n_boot=200)
    assert set(summary) == set(metrics.METRICS)
    assert summary["valid_json"]["value"] == pytest.approx(0.5)
    for m in summary.values():
        assert m["lo"] <= m["value"] <= m["hi"]


def test_bootstrap_is_seeded_and_paired_difference_of_identical_runs_is_zero():
    items = [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0]
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    assert interval(items, mean, n_boot=300, seed=1) == interval(items, mean, n_boot=300, seed=1)
    keyed = [(f"r{i}", x) for i, x in enumerate(items)]
    diff = paired_difference(
        keyed, keyed, lambda xs: mean([x for _, x in xs]), key=lambda kx: kx[0], n_boot=300
    )
    assert diff == (0.0, 0.0, 0.0)


def test_paired_difference_needs_the_same_receipts():
    with pytest.raises(ValueError):
        paired_difference([("a", 1.0)], [("b", 1.0)], lambda xs: 0.0, key=lambda kx: kx[0])
