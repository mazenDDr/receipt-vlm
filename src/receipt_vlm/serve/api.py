"""An HTTP API around the inference runners: one receipt image in, its fields out.

    <env>/bin/uvicorn "receipt_vlm.serve.api:default_app" --factory --port 8000

The runtime comes from `configs/infer.yaml` like every other run, so the same service can front
transformers, vLLM (AWQ W4A16) or llama.cpp (GGUF) by changing `backend` and `model` - no code
changes. Every response carries the per-request readout the deployment decision was made on:
latency, token counts and throughput.

fastapi is imported inside the functions, the same way `infer/runner.py` imports torch, so this
module can be imported and the rest of the test suite can run without the `serve` extra installed.
"""

# No `from __future__ import annotations` in this module, deliberately. FastAPI resolves route
# annotations at runtime against the module's globals; with postponed annotations the names imported
# inside create_app (UploadFile, File) are invisible there, so `file` was read as a JSON body field
# and every multipart upload failed validation with 422 before the handler ran.
import hashlib
import os
import tempfile
import uuid
from pathlib import Path
from typing import Annotated, Any

from receipt_vlm.infer import backends
from receipt_vlm.infer.runner import InferConfig
from receipt_vlm.schemas import Prediction, ReceiptExample

# What a browser or `curl -F` will send. The runners re-encode nothing, so these pass straight through.
IMAGE_SUFFIX = {"image/png": ".png", "image/jpeg": ".jpg", "image/jpg": ".jpg", "image/webp": ".webp"}


def example_for_upload(path: str | Path, data: bytes) -> ReceiptExample:
    """A ReceiptExample standing in for an uploaded receipt.

    All three runners read only `example_id` and `image_path`, so the gold fields carry no meaning
    here: `target` is empty and `n_fields` is 0. `split` is required by the shared contract and is
    never read for a served request. The image facts are real, not placeholders.
    """
    from PIL import Image

    with Image.open(path) as image:
        width, height = image.size
    return ReceiptExample(
        example_id=f"upload-{uuid.uuid4().hex[:12]}",
        split="test",  # unread by every runner; a served receipt belongs to no split
        image_path=str(path),
        image_sha256=hashlib.sha256(data).hexdigest(),
        width=width,
        height=height,
        target={},  # no gold label for a served image
        n_fields=0,
    )


def readout(prediction: Prediction, gpu_cost_per_hour: float = 0.0) -> dict[str, Any]:
    """The per-request numbers the deployment choice was made on (see docs/tradeoffs.md)."""
    latency = prediction.latency_s
    out: dict[str, Any] = {
        "valid_json": prediction.parsed is not None,
        "truncated": prediction.truncated,
        "latency_s": round(latency, 3),
        "prompt_tokens": prediction.prompt_tokens,
        "output_tokens": prediction.output_tokens,
        "output_tokens_per_s": round(prediction.output_tokens / latency, 1) if latency else 0.0,
    }
    if gpu_cost_per_hour > 0:
        # Wall clock only: this machine is not shared, so a request owns the GPU while it runs.
        out["estimated_cost_usd"] = round(latency / 3600 * gpu_cost_per_hour, 6)
    return out


def run_one(
    engine: Any, cfg: InferConfig, model: Any, processor: Any, data: bytes, suffix: str
) -> Prediction:
    """Write the upload to a temp file, run one receipt through the engine, clean up."""
    handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)  # noqa: SIM115 (closed below)
    try:
        handle.write(data)
        handle.close()
        example = example_for_upload(handle.name, data)
        return next(iter(engine.predict([example], cfg, model, processor)))
    finally:
        os.unlink(handle.name)


def create_app(cfg: InferConfig, engine: Any = None, gpu_cost_per_hour: float = 0.0) -> Any:
    """The FastAPI app. `engine` is injectable so the tests run on a machine with no model."""
    from fastapi import FastAPI, File, HTTPException, UploadFile

    app = FastAPI(title="receipt-vlm", description="Receipt field extraction from a photo.")
    state: dict[str, Any] = {
        "engine": engine if engine is not None else backends.get(cfg.backend),
        "model": None,
        "processor": None,
        "loaded": False,
    }

    def ensure_loaded() -> tuple[Any, Any]:
        """Load on first request rather than at import, so /health answers while the model loads."""
        if not state["loaded"]:
            state["model"], state["processor"] = state["engine"].load(cfg)
            state["loaded"] = True
        return state["model"], state["processor"]

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "variant": cfg.variant,
            "backend": cfg.backend,
            "model_loaded": state["loaded"],
        }

    @app.post("/extract")
    async def extract(file: Annotated[UploadFile, File()]) -> dict[str, Any]:
        suffix = IMAGE_SUFFIX.get((file.content_type or "").lower())
        if suffix is None:
            raise HTTPException(415, f"unsupported type {file.content_type!r}; send PNG, JPEG or WebP")
        data = await file.read()
        if not data:
            raise HTTPException(400, "empty upload")
        model, processor = ensure_loaded()
        prediction = run_one(state["engine"], cfg, model, processor, data, suffix)
        return {
            "fields": prediction.parsed,
            "raw_output": prediction.raw_output,
            "variant": cfg.variant,
            "backend": cfg.backend,
            **readout(prediction, gpu_cost_per_hour),
        }

    return app


def default_app() -> Any:
    """Entry point for `uvicorn "receipt_vlm.serve.api:default_app" --factory`.

    RECEIPT_VLM_CONFIG picks the runtime (default configs/infer.yaml);
    RECEIPT_VLM_GPU_COST_PER_HOUR adds an estimated cost to each response when set.
    """
    from receipt_vlm import config

    cfg = config.load(Path(os.environ.get("RECEIPT_VLM_CONFIG", "configs/infer.yaml")), InferConfig)
    rate = float(os.environ.get("RECEIPT_VLM_GPU_COST_PER_HOUR", "0"))
    return create_app(cfg, gpu_cost_per_hour=rate)
