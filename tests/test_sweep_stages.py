from receipt_vlm.train import sweep


def test_stage_b_varies_only_the_rank_at_one_epoch():
    arms = sweep.load_arms("configs/sweeps/stage-b-rank.yaml")
    assert [(a.lora_r, a.lora_alpha) for a in arms] == [(16, 32), (8, 16), (32, 64)]
    same = {(a.learning_rate, a.epochs, a.eval_limit, a.targets, a.precision, a.max_pixels) for a in arms}
    assert same == {(4e-4, 1.0, 0, "lm", "nf4", 802816)}
