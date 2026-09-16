"""The serving layer, exercised with a fake engine so these run on the Mac with no model."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from receipt_vlm.infer.runner import InferConfig
from receipt_vlm.schemas import Prediction
from receipt_vlm.serve.api import example_for_upload, readout
from receipt_vlm.serve.gradio_app import format_readout

fastapi = pytest.importorskip("fastapi", reason="needs the 'serve' extra")
from fastapi.testclient import TestClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "tests/fixtures/cord_mini/images/cord-test-0091.png"


_UNSET = object()  # None is a meaningful value for `parsed`: the output was not valid JSON


def fake_engine(parsed=_UNSET, raw='{"total": {"total_price": "21.000"}}', truncated=False):
    """Stands in for a runner module: the same load/predict surface, no model."""
    calls = {"load": 0, "images": []}

    def load(cfg):
        calls["load"] += 1
        return "model", "processor"

    def predict(examples, cfg, model, processor):
        for example in examples:
            calls["images"].append(Path(example.image_path).read_bytes())
            yield Prediction(
                example_id=example.example_id,
                variant=cfg.variant,
                raw_output=raw,
                parsed={"total": {"total_price": "21.000"}} if parsed is _UNSET else parsed,
                prompt_tokens=1270,
                output_tokens=55,
                latency_s=1.5,
                truncated=truncated,
            )

    return SimpleNamespace(load=load, predict=predict), calls


def client(engine, **kwargs):
    from receipt_vlm.serve.api import create_app

    cfg = InferConfig(variant="ft-r16-awq-w4a16", backend="vllm")
    return TestClient(create_app(cfg, engine=engine, **kwargs))


def upload(name=RECEIPT, content_type="image/png"):
    return {"file": (name.name, name.read_bytes(), content_type)}


def test_health_reports_the_variant_and_whether_the_model_is_loaded_yet():
    engine, _ = fake_engine()
    with client(engine) as api:
        body = api.get("/health").json()
    assert body["status"] == "ok"
    assert body["variant"] == "ft-r16-awq-w4a16" and body["backend"] == "vllm"
    assert body["model_loaded"] is False  # /health must answer while the model is still loading


def test_extract_returns_the_fields_and_the_per_request_readout():
    engine, _ = fake_engine()
    with client(engine) as api:
        body = api.post("/extract", files=upload()).json()
    assert body["fields"] == {"total": {"total_price": "21.000"}}
    assert body["valid_json"] is True and body["truncated"] is False
    assert body["prompt_tokens"] == 1270 and body["output_tokens"] == 55
    assert body["latency_s"] == 1.5 and body["output_tokens_per_s"] == pytest.approx(36.7, abs=0.1)
    assert body["variant"] == "ft-r16-awq-w4a16" and body["backend"] == "vllm"
    assert "estimated_cost_usd" not in body  # no rate configured, so no invented cost


def test_the_model_loads_once_and_the_upload_reaches_the_engine_unchanged():
    engine, calls = fake_engine()
    with client(engine) as api:
        api.post("/extract", files=upload())
        api.post("/extract", files=upload())
    assert calls["load"] == 1
    assert calls["images"] == [RECEIPT.read_bytes()] * 2  # bytes pass through, never re-encoded


def test_a_cost_appears_only_when_a_rate_is_configured():
    engine, _ = fake_engine()
    with client(engine, gpu_cost_per_hour=1.8) as api:
        body = api.post("/extract", files=upload()).json()
    assert body["estimated_cost_usd"] == pytest.approx(1.5 / 3600 * 1.8, abs=1e-9)


def test_output_that_is_not_json_still_answers_with_valid_json_false():
    engine, _ = fake_engine(parsed=None, raw="sorry, I cannot read this receipt")
    with client(engine) as api:
        response = api.post("/extract", files=upload())
    body = response.json()
    assert response.status_code == 200  # the model ran; the caller decides what to do
    assert body["fields"] is None and body["valid_json"] is False
    assert body["raw_output"] == "sorry, I cannot read this receipt"


@pytest.mark.parametrize(
    ("files", "status"),
    [
        ({"file": ("notes.txt", b"not an image", "text/plain")}, 415),
        ({"file": ("empty.png", b"", "image/png")}, 400),
    ],
)
def test_bad_uploads_are_refused_before_the_model_is_touched(files, status):
    engine, calls = fake_engine()
    with client(engine) as api:
        assert api.post("/extract", files=files).status_code == status
    assert calls["load"] == 0


def test_a_served_upload_carries_real_image_facts_and_empty_gold_fields():
    data = RECEIPT.read_bytes()
    example = example_for_upload(RECEIPT, data)
    assert example.example_id.startswith("upload-")
    assert example.width > 0 and example.height > 0
    assert example.image_sha256 == __import__("hashlib").sha256(data).hexdigest()
    assert example.target == {} and example.n_fields == 0  # a served receipt has no gold label


def test_the_readout_reports_throughput_and_survives_a_zero_latency():
    prediction = Prediction(
        example_id="upload-x",
        variant="v",
        raw_output="{}",
        parsed={},
        prompt_tokens=10,
        output_tokens=100,
        latency_s=2.0,
    )
    assert readout(prediction)["output_tokens_per_s"] == 50.0
    stalled = prediction.model_copy(update={"latency_s": 0.0})
    assert readout(stalled)["output_tokens_per_s"] == 0.0  # must not divide by zero


def test_the_demo_readout_line_names_the_runtime_and_flags_bad_output():
    numbers = {
        "valid_json": False,
        "truncated": True,
        "latency_s": 1.51,
        "prompt_tokens": 1270,
        "output_tokens": 55,
        "output_tokens_per_s": 36.7,
        "estimated_cost_usd": 0.00075,
    }
    line = format_readout(numbers, "ft-r16-gguf-q4_k_m", "llamacpp")
    assert "ft-r16-gguf-q4_k_m" in line and "llamacpp" in line
    assert "1.51 s" in line and "55 output tokens" in line
    assert "not valid JSON" in line and "truncated" in line and "$0.00075" in line
