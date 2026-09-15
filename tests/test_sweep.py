import pytest

from receipt_vlm.train import sweep

BASE = "base: configs/train/qlora-r16-lm.yaml\narms:\n"


def _sweep_file(tmp_path, arms: str):
    path = tmp_path / "stage.yaml"
    path.write_text(BASE + arms)
    return path


def test_arms_override_the_base_config(tmp_path):
    arms = sweep.load_arms(
        _sweep_file(tmp_path, "  - {variant: a}\n  - {variant: b, learning_rate: 0.0001, lora_r: 8}\n")
    )
    assert [a.variant for a in arms] == ["a", "b"]
    assert arms[0].learning_rate == 2e-4
    assert arms[1].learning_rate == 1e-4 and arms[1].lora_r == 8 and arms[1].precision == "nf4"


def test_a_typo_in_a_setting_name_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="learning_rat"):
        sweep.load_arms(_sweep_file(tmp_path, "  - {variant: c, learning_rat: 0.1}\n"))


def test_arms_need_their_own_unique_variant_names(tmp_path):
    with pytest.raises(ValueError, match="variant name"):
        sweep.load_arms(_sweep_file(tmp_path, "  - {learning_rate: 0.0001}\n"))
    with pytest.raises(ValueError, match="twice"):
        sweep.load_arms(_sweep_file(tmp_path, "  - {variant: d}\n  - {variant: d, lora_r: 8}\n"))


def test_is_trained_looks_for_an_adapter_or_full_model(tmp_path):
    (tmp_path / "v1" / "adapter").mkdir(parents=True)
    (tmp_path / "v2" / "checkpoints").mkdir(parents=True)
    assert sweep.is_trained("v1", tmp_path)
    assert not sweep.is_trained("v2", tmp_path) and not sweep.is_trained("v3", tmp_path)


def test_stage_a_sweeps_the_learning_rate_only():
    arms = sweep.load_arms("configs/sweeps/stage-a-lr.yaml")
    assert [a.learning_rate for a in arms] == [2e-4, 1e-4, 4e-4]
    assert {(a.lora_r, a.targets, a.precision) for a in arms} == {(16, "lm", "nf4")}
