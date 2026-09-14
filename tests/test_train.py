import re

import pytest

from receipt_vlm import config
from receipt_vlm.train import collate, qlora

HEADER = [151644, 77091, 198]  # <|im_start|> assistant \n


def test_answer_labels_keep_only_the_answer_after_the_last_header():
    #        system / user turn ...        header ...   answer + <|im_end|>   padding
    ids = [1, 151644, 77091, 198, 5, 6, 7] + HEADER + [40, 41, 151645] + [0, 0]
    mask = [1] * 13 + [0, 0]
    labels = collate.answer_labels(ids, mask, HEADER)
    assert labels == [collate.IGNORE] * 10 + [40, 41, 151645] + [collate.IGNORE] * 2


def test_answer_labels_refuse_a_sequence_without_an_answer():
    with pytest.raises(ValueError):
        collate.answer_labels([1, 2, 3], [1, 1, 1], HEADER)


@pytest.mark.parametrize(
    ("name", "lm", "lm_merger"),
    [
        ("model.language_model.layers.0.self_attn.q_proj", True, True),
        ("model.language_model.layers.35.mlp.down_proj", True, True),
        # same leaf names as the language model, but inside the frozen vision encoder
        ("model.visual.blocks.3.mlp.down_proj", False, False),
        ("model.visual.blocks.3.attn.qkv", False, False),
        ("model.visual.merger.mlp.0", False, True),
        ("model.visual.merger.mlp.2", False, True),
        ("lm_head", False, False),
    ],
)
def test_lora_targets_stay_out_of_the_vision_encoder(name, lm, lm_merger):
    assert bool(re.fullmatch(qlora.lora_target_regex("lm"), name)) is lm
    assert bool(re.fullmatch(qlora.lora_target_regex("lm+merger"), name)) is lm_merger


def test_expected_wrapped_matches_the_measured_counts():
    # measured on the real module tree (meta device): 252 for lm, 254 for lm+merger
    assert qlora.expected_wrapped(36, "lm") == 252
    assert qlora.expected_wrapped(36, "lm+merger") == 254


def test_train_config_file_loads_with_the_personal_wandb_entity():
    cfg = config.load("configs/train/qlora-r16-lm.yaml", qlora.TrainConfig)
    assert cfg.precision == "nf4" and cfg.targets == "lm" and cfg.wandb_entity == "khaledmazen456"
