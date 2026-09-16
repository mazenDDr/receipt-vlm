"""A Gradio demo over the same runners as the API: drop in a receipt, see the fields and the cost.

    <env>/bin/python -m receipt_vlm.serve.gradio_app

gradio is imported inside the function, so importing this module does not require the `demo` extra.
The heavy lifting is shared with `serve/api.py`, so the demo and the endpoint cannot drift apart.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

from receipt_vlm.infer import backends
from receipt_vlm.infer.runner import InferConfig
from receipt_vlm.serve.api import readout, run_one

DESCRIPTION = """Upload a receipt photo and the model returns its fields as JSON: menu items with
counts and prices, subtotal, tax, total and payment. Fine-tuned from Qwen2.5-VL-3B on CORD v2."""


def format_readout(numbers: dict[str, Any], variant: str, backend: str) -> str:
    """The per-request line under the JSON: what this answer cost to produce."""
    parts = [
        f"**{variant}** on **{backend}**",
        f"{numbers['latency_s']:.2f} s",
        f"{numbers['output_tokens']} output tokens ({numbers['output_tokens_per_s']:.0f}/s)",
        f"{numbers['prompt_tokens']} prompt tokens",
    ]
    if not numbers["valid_json"]:
        parts.append("**output was not valid JSON**")
    if numbers["truncated"]:
        parts.append("**truncated at the token cap**")
    if "estimated_cost_usd" in numbers:
        parts.append(f"~${numbers['estimated_cost_usd']:.5f}")
    return " · ".join(parts)


def build_demo(cfg: InferConfig, engine: Any = None, gpu_cost_per_hour: float = 0.0) -> Any:
    """The Gradio interface. `engine` is injectable, exactly as in `create_app`."""
    import gradio as gr

    state: dict[str, Any] = {
        "engine": engine if engine is not None else backends.get(cfg.backend),
        "model": None,
        "processor": None,
        "loaded": False,
    }

    def extract(image: Any) -> tuple[dict[str, Any] | None, str]:
        if image is None:
            return None, "Upload a receipt first."
        if not state["loaded"]:
            state["model"], state["processor"] = state["engine"].load(cfg)
            state["loaded"] = True
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        prediction = run_one(
            state["engine"], cfg, state["model"], state["processor"], buffer.getvalue(), ".png"
        )
        numbers = readout(prediction, gpu_cost_per_hour)
        fields = prediction.parsed if prediction.parsed is not None else {"raw_output": prediction.raw_output}
        return fields, format_readout(numbers, cfg.variant, cfg.backend)

    with gr.Blocks(title="receipt-vlm") as demo:
        gr.Markdown("# Receipt field extraction")
        gr.Markdown(DESCRIPTION)
        with gr.Row():
            image_in = gr.Image(type="pil", label="Receipt", sources=["upload", "clipboard"])
            with gr.Column():
                fields_out = gr.JSON(label="Extracted fields")
                cost_out = gr.Markdown()
        gr.Button("Extract", variant="primary").click(extract, image_in, [fields_out, cost_out])
    return demo


def main() -> None:
    from receipt_vlm import config

    cfg = config.load(Path(os.environ.get("RECEIPT_VLM_CONFIG", "configs/infer.yaml")), InferConfig)
    rate = float(os.environ.get("RECEIPT_VLM_GPU_COST_PER_HOUR", "0"))
    build_demo(cfg, gpu_cost_per_hour=rate).launch(server_name="0.0.0.0", server_port=7860)  # noqa: S104


if __name__ == "__main__":
    main()
