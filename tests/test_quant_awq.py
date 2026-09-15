import pytest

from receipt_vlm.quant import awq


def _names(n_layers=2, n_vision=2):
    lm = [
        f"model.language_model.layers.{i}.{leaf}"
        for i in range(n_layers)
        for leaf in (
            "input_layernorm",
            "post_attention_layernorm",
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        )
    ]
    vision = [
        f"model.visual.blocks.{i}.{leaf}"
        for i in range(n_vision)
        for leaf in (
            "norm1",
            "norm2",
            "attn.qkv",
            "attn.proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        )
    ]
    return lm + vision + ["model.visual.merger.mlp.0", "model.visual.merger.mlp.2", "lm_head"]


def test_the_built_in_style_pattern_reaches_into_the_vision_tower():
    # llm-compressor's registry uses "re:.*up_proj$": it also hits the vision MLPs
    assert len(awq.matches(r"re:.*up_proj$", _names())) == 4


def test_anchored_mappings_hit_one_module_per_language_layer_and_no_vision_module():
    awq.check_mappings(_names(n_layers=2), n_layers=2)
    for smooth, balance in awq.MAPPINGS:
        for pattern in (smooth, *balance):
            assert not [n for n in awq.matches(pattern, _names()) if "visual" in n]


def test_check_mappings_fails_when_the_layer_count_is_off():
    with pytest.raises(RuntimeError, match="expected 3"):
        awq.check_mappings(_names(n_layers=2), n_layers=3)


def test_ignore_keeps_the_vision_side_and_lm_head_out_of_quantization():
    names = _names()
    ignored = {n for pattern in awq.IGNORE for n in awq.matches(pattern, names)}
    assert "lm_head" in ignored and "model.visual.merger.mlp.0" in ignored
    assert all(n in ignored for n in names if "visual" in n)
    assert not any("language_model" in n for n in ignored)


class _Module:
    def __init__(self, scheme):
        self.quantization_scheme = scheme


class _Model:
    def __init__(self, names):
        self._names = names

    def named_modules(self):
        return [(n, _Module("w4a16" if q else None)) for n, q in self._names]


def test_check_quantized_wants_exactly_the_language_linears():
    lm_linears = [f"model.language_model.layers.0.{p}" for p in ("q", "k", "v", "o", "gate", "up", "down")]
    good = _Model([(n, True) for n in lm_linears] + [("model.visual.blocks.0.mlp.up_proj", False)])
    awq.check_quantized(good, n_layers=1)
    bad = _Model([(n, True) for n in lm_linears] + [("model.visual.blocks.0.mlp.up_proj", True)])
    with pytest.raises(RuntimeError, match="outside the LM"):
        awq.check_quantized(bad, n_layers=1)
