import pytest

from receipt_vlm.eval import fields
from receipt_vlm.eval.parse import extract_json
from receipt_vlm.eval.report import render, run_stats
from receipt_vlm.infer import runner
from receipt_vlm.prompts import INSTRUCTION, build_messages, serialize_target
from receipt_vlm.schemas import Prediction

TRICKY = {
    "menu": [
        {"nm": "Café Latte", "cnt": "1", "price": "28,000", "sub": [{"nm": "Less Ice"}, {"nm": "Less Ice"}]},
        {"nm": "TEA", "price": "8.000"},
    ],
    "total": {"cashprice": ["50,000", "10,000"], "total_price": "36,000"},
}


def test_instruction_names_every_cord_key():
    for path in fields.TEXT_KEYS | fields.NUMERIC_KEYS:
        top, *_, last = path.split(".")
        assert f'"{top}"' in INSTRUCTION and f'"{last}"' in INSTRUCTION, path


def test_training_target_parses_back_to_the_gold_receipt():
    assert extract_json(serialize_target(TRICKY)) == TRICKY


def test_build_messages_with_and_without_target():
    prompt_only = build_messages()
    assert [m["role"] for m in prompt_only] == ["user"]
    assert prompt_only[0]["content"][0] == {"type": "image"}
    with_target = build_messages(TRICKY)
    assert with_target[-1]["role"] == "assistant"
    assert with_target[-1]["content"][0]["text"] == serialize_target(TRICKY)


def test_generated_tokens_stop_at_the_first_eos():
    assert runner.generated_tokens([5, 6, 99, 7, 99], {99}) == [5, 6, 99]
    assert runner.generated_tokens([5, 6, 7], {99}) == [5, 6, 7]


def test_image_cap_check_catches_an_ignored_cap():
    runner.check_image_cap([[1, 38, 26]], patch_size=14, max_pixels=256 * 28 * 28)
    with pytest.raises(RuntimeError):
        runner.check_image_cap([[1, 92, 62]], patch_size=14, max_pixels=256 * 28 * 28)


def test_image_kwargs_use_the_size_form():
    assert runner.image_kwargs(3136, 200704) == {"size": {"shortest_edge": 3136, "longest_edge": 200704}}


def test_bnb_keeps_lm_head_and_by_default_the_vision_tower_in_bf16():
    assert runner.bnb_skip_modules(quantize_vision=False) == ["lm_head", "model.visual"]
    assert runner.bnb_skip_modules(quantize_vision=True) == ["lm_head"]


def test_run_stats_and_render():
    preds = [
        Prediction(
            example_id=f"x{i}",
            variant="v",
            raw_output="{}",
            parsed={},
            prompt_tokens=300,
            output_tokens=100 + i,
            latency_s=2.0,
            peak_vram_mb=8000.0,
            truncated=i == 0,
        )
        for i in range(4)
    ]
    stats = run_stats(preds)
    assert stats["truncated"] == 0.25 and stats["output_tokens_per_s"] == pytest.approx(406 / 8)
    summary = {
        "variant": "v",
        "split": "dev",
        "n": 4,
        "metrics": {"f1": {"value": 0.5, "lo": 0.4, "hi": 0.6}},
        "run": stats,
    }
    text = render(summary)
    assert "| Field F1 | 0.500 | [0.400, 0.600] |" in text and "25.0%" in text
    assert stats["peak_vram_mb"] == 8000.0 and "| Peak VRAM (MB) | 8000 |" in text


def test_unmeasured_peak_vram_is_a_dash_not_a_zero():
    """vLLM and llama.cpp hold memory in another process, so their runs record no peak.

    The first version summed `p.peak_vram_mb or 0.0`, and every out-of-process run reported a peak of
    "0 MB" in its committed report — a measured zero, which is not what happened.
    """
    preds = [
        Prediction(
            example_id=f"cord-validation-{i:04d}",
            variant="ft-r16-gguf-q8_0",
            raw_output="{}",
            parsed={},
            prompt_tokens=1234,
            output_tokens=97,
            latency_s=1.3,
            peak_vram_mb=None,
            truncated=False,
        )
        for i in range(3)
    ]
    stats = run_stats(preds)
    assert stats["peak_vram_mb"] is None
    summary = {"variant": "v", "split": "dev", "n": 3, "metrics": {}, "run": stats}
    assert "| Peak VRAM (MB) | — |" in render(summary)
