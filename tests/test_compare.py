import pytest

from receipt_vlm.eval import compare
from receipt_vlm.schemas import ExampleScore, FieldCounts


def _run(tmp_path, name, tps):
    """A fake run dir whose receipt i found tps[i] of 10 fields."""
    run = tmp_path / name
    run.mkdir()
    scores = [
        ExampleScore(
            example_id=f"r{i}",
            variant=name,
            valid_json=True,
            exact_match=tp == 10,
            overall=FieldCounts(tp=tp, fp=10 - tp, fn=10 - tp),
            numeric=FieldCounts(tp=tp, fp=10 - tp, fn=10 - tp),
            text=FieldCounts(),
            ted_accuracy=tp / 10,
        )
        for i, tp in enumerate(tps)
    ]
    (run / "scores.jsonl").write_text("".join(s.model_dump_json() + "\n" for s in scores))
    return run


def test_a_clear_improvement_is_real_and_identical_runs_are_not(tmp_path):
    better = _run(tmp_path, "ft", [9, 10, 8, 9, 10, 9, 8, 10, 9, 9] * 3)
    worse = _run(tmp_path, "base", [5, 6, 4, 5, 6, 5, 4, 6, 5, 5] * 3)
    diffs = compare.compare(better, worse, n_boot=300)
    f1, lo, hi = diffs["f1"]
    assert f1 == pytest.approx(0.91 - 0.51) and lo > 0 and compare.is_real(diffs["f1"])
    same = compare.compare(better, better, n_boot=300)
    assert same["f1"] == (0.0, 0.0, 0.0) and not compare.is_real(same["f1"])


def test_render_marks_real_differences(tmp_path):
    text = compare.render("ft", "base", {"f1": (0.4, 0.35, 0.45), "ted_accuracy": (0.01, -0.02, 0.04)}, 30)
    assert "| Field F1 | +0.400 | [+0.350, +0.450] | yes |" in text
    assert "| TED accuracy | +0.010 | [-0.020, +0.040] | no |" in text
