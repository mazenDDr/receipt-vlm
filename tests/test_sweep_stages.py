from receipt_vlm.train import sweep


def test_stage_b_varies_only_the_rank_at_one_epoch():
    arms = sweep.load_arms("configs/sweeps/stage-b-rank.yaml")
    assert [(a.lora_r, a.lora_alpha) for a in arms] == [(16, 32), (8, 16), (32, 64)]
    same = {(a.learning_rate, a.epochs, a.eval_limit, a.targets, a.precision, a.max_pixels) for a in arms}
    assert same == {(4e-4, 1.0, 0, "lm", "nf4", 802816)}


def test_stage_cde_changes_one_thing_per_arm_against_the_stage_b_reference():
    arms = {a.variant: a for a in sweep.load_arms("configs/sweeps/stage-cde.yaml")}
    c, d = arms["qlora-r16-lmproj-lr4e4-e1"], arms["lora-bf16-r16-lm-lr4e4-e1"]
    assert (c.targets, c.precision, c.method, c.learning_rate) == ("lm+merger", "nf4", "lora", 4e-4)
    assert (d.targets, d.precision, d.method, d.learning_rate) == ("lm", "bf16", "lora", 4e-4)
    partial = [a for a in arms.values() if a.method == "partial"]
    assert sorted(a.learning_rate for a in partial) == [1e-5, 5e-5]
    assert {(a.precision, a.optim, a.partial_last_n_layers) for a in partial} == {
        ("bf16", "paged_adamw_8bit", 4)
    }
    assert all(a.epochs == 1 and a.eval_limit == 0 and a.lora_r == 16 for a in arms.values())
    assert {a.cuda_alloc_conf for a in arms.values() if a.precision == "bf16"} == {"expandable_segments:True"}
    assert c.cuda_alloc_conf is None
