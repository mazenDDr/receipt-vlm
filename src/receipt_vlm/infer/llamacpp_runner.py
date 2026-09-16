"""The same prediction loop on llama.cpp, for the GGUF bit widths.

`llama-server` is started with the text GGUF and the vision projector, and each receipt goes through its
OpenAI-style chat endpoint with the same prompt, image cap and greedy decoding as the other runners. Only the
standard library is used for HTTP, so no env needs extra packages.
"""

from __future__ import annotations

import base64
import json
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from receipt_vlm.eval.parse import extract_json
from receipt_vlm.infer.runner import InferConfig
from receipt_vlm.prompts import INSTRUCTION
from receipt_vlm.schemas import Prediction, ReceiptExample


def server_command(cfg: InferConfig) -> list[str]:
    """llama-server with everything pinned: same image cap, greedy decoding, all layers on the GPU."""
    if not cfg.mmproj:
        raise ValueError("GGUF runs need --mmproj (the vision projector)")
    return [
        cfg.llama_server,
        "-m",
        cfg.model,
        "--mmproj",
        cfg.mmproj,
        "-ngl",
        str(cfg.n_gpu_layers),
        "--image-max-tokens",
        str(cfg.image_max_tokens),
        "-c",
        str(cfg.max_model_len),
        # One slot, so the whole context belongs to this request. Left to "auto", llama-server splits n_ctx
        # across slots, and a slot smaller than the ~1300-token prompt cannot hold a receipt.
        "-np",
        "1",
        "--temp",
        "0",
        "--top-k",
        "1",
        "--repeat-penalty",
        str(cfg.repetition_penalty),
        "--port",
        str(cfg.server_port),
        "--host",
        "127.0.0.1",
        "--no-warmup",
        # Belt and braces with the per-request cache_prompt=false: same text prompt, different image every
        # time, and a reused prefix produced 1024 "!" tokens from the fifth receipt on.
        "--no-cache-prompt",
        "--no-context-shift",
    ]


def chat_request(image_bytes: bytes, media_type: str, cfg: InferConfig) -> dict[str, Any]:
    """The same turn as build_messages(): the image first, then the instruction."""
    data_url = f"data:{media_type};base64,{base64.b64encode(image_bytes).decode()}"
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": INSTRUCTION},
                ],
            }
        ],
        "temperature": 0.0,
        "top_k": 1,
        "max_tokens": cfg.max_new_tokens,
        "seed": 0,
        # Every receipt shares the same text prompt and differs only in its image. With prompt caching on,
        # the server reused the cached prefix and from the fifth receipt on emitted 1024 "!" tokens — the
        # symptom of an image embedding that was never filled in. Each receipt gets a clean prompt.
        "cache_prompt": False,
    }


def is_truncated(finish_reason: str | None) -> bool:
    return finish_reason == "length"


def media_type(path: str | Path) -> str:
    return "image/png" if str(path).lower().endswith(".png") else "image/jpeg"


def _post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 (localhost only)
        return json.loads(response.read())


def server_log_path(variant: str) -> Path:
    return Path("outputs") / f"llama-server-{variant}.log"


def load(cfg: InferConfig) -> tuple[Any, Any]:
    """Start llama-server and wait until it answers /health.

    The server's own output goes to a log file, never to DEVNULL: when the model emitted floods of "!"
    tokens, its stderr was the one place that said why, and it had been thrown away.
    """
    log_path = server_log_path(cfg.variant)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w")  # noqa: SIM115 (closed when the server process ends)
    process = subprocess.Popen(server_command(cfg), stdout=log_file, stderr=subprocess.STDOUT)  # noqa: S603
    health = f"http://127.0.0.1:{cfg.server_port}/health"
    deadline = time.time() + cfg.server_startup_s
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited with {process.returncode}; see {log_path}")
        try:
            with urllib.request.urlopen(health, timeout=5) as response:  # noqa: S310
                if response.status == 200:
                    return process, None
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(2)
    process.terminate()
    raise RuntimeError(f"llama-server unhealthy after {cfg.server_startup_s}s; see {log_path}")


def predict(
    examples: Sequence[ReceiptExample], cfg: InferConfig, model: Any, processor: Any
) -> Iterator[Prediction]:
    url = f"http://127.0.0.1:{cfg.server_port}/v1/chat/completions"
    try:
        for example in examples:
            image_bytes = Path(example.image_path).read_bytes()
            payload = chat_request(image_bytes, media_type(example.image_path), cfg)
            t0 = time.perf_counter()
            answer = _post(url, payload, timeout=cfg.request_timeout_s)
            latency = time.perf_counter() - t0
            choice = answer["choices"][0]
            text = choice["message"]["content"]
            usage = answer.get("usage", {})
            yield Prediction(
                example_id=example.example_id,
                variant=cfg.variant,
                raw_output=text,
                parsed=extract_json(text),
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                output_tokens=int(usage.get("completion_tokens", 0)),
                latency_s=latency,
                peak_vram_mb=None,  # llama-server runs in its own process; memory is measured separately
                truncated=is_truncated(choice.get("finish_reason")),
            )
    finally:
        model.terminate()
        model.wait(timeout=30)
