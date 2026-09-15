import json

import pytest

from receipt_vlm.eval import tables


def _run(tmp_path, name, split="dev", f1=0.5, train=None):
    run = tmp_path / name
    run.mkdir()
    metric = {"value": f1, "lo": f1 - 0.04, "hi": f1 + 0.04}
    summary = {"variant": name, "split": split, "n": 99, "metrics": dict.fromkeys(tables.METRIC_KEYS, metric)}
    (run / "summary.json").write_text(json.dumps(summary))
    if train:
        (run / "train_summary.json").write_text(json.dumps(train))
    return run


def test_rows_show_training_cost_only_for_trained_runs(tmp_path):
    base = _run(tmp_path, "base-bf16", f1=0.494)
    trained = {"train_seconds": 3000, "train_peak_vram_mb": 12288, "trainable_params": 29_900_000}
    ft = _run(tmp_path, "qlora-r16-lm", f1=0.9, train=trained)
    text = tables.render([tables.run_row(base), tables.run_row(ft)])
    base_row, ft_row = text.splitlines()[2:4]
    assert base_row.startswith("| base-bf16 | 99 | 0.494 [0.454, 0.534] |")
    assert base_row.endswith("| — | — | — |") and ft_row.endswith("| 50 | 12.0 | 29.9 |")


def test_an_unrecorded_peak_shows_as_a_dash_not_zero(tmp_path):
    recovered = {"train_seconds": 3910.6, "train_peak_vram_mb": None, "trainable_params": 29_933_568}
    row = tables.render([tables.run_row(_run(tmp_path, "qlora-r16-lm", train=recovered))]).splitlines()[2]
    assert row.endswith("| 65 | — | 29.9 |")


def test_a_table_refuses_to_mix_dev_and_test(tmp_path):
    rows = [tables.run_row(_run(tmp_path, "a")), tables.run_row(_run(tmp_path, "b", split="test"))]
    with pytest.raises(ValueError, match="different splits"):
        tables.render(rows)
