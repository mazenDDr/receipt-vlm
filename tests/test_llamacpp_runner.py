import ast
import base64
import json
import re
from pathlib import Path
from typing import get_args

import pytest

from receipt_vlm.infer import backends, llamacpp_runner, runner
from receipt_vlm.prompts import INSTRUCTION


def _cfg(**kwargs):
    return runner.InferConfig(model="models/x/model-Q4_K_M.gguf", mmproj="models/x/mmproj-f16.gguf", **kwargs)


def _flags(command: list[str]) -> tuple[dict[str, str], set[str]]:
    """Split the command into flags that take a value and flags that stand alone."""
    pairs: dict[str, str] = {}
    bare: set[str] = set()
    i = 1  # command[0] is the binary
    while i < len(command):
        token = command[i]
        if i + 1 < len(command) and not command[i + 1].startswith("-"):
            pairs[token] = command[i + 1]
            i += 2
        else:
            bare.add(token)
            i += 1
    return pairs, bare


def test_the_server_gets_the_same_image_cap_and_greedy_settings():
    pairs, bare = _flags(llamacpp_runner.server_command(_cfg(image_max_tokens=1024, max_model_len=4096)))
    assert pairs["-m"] == "models/x/model-Q4_K_M.gguf" and pairs["--mmproj"] == "models/x/mmproj-f16.gguf"
    assert pairs["--image-max-tokens"] == "1024" and pairs["-c"] == "4096"
    assert pairs["--temp"] == "0" and pairs["--top-k"] == "1" and pairs["--repeat-penalty"] == "1.0"
    assert pairs["-ngl"] == "99" and pairs["-np"] == "1"
    assert {"--no-cache-prompt", "--no-context-shift"} <= bare


def test_a_gguf_run_without_the_vision_projector_is_refused():
    with pytest.raises(ValueError, match="mmproj"):
        llamacpp_runner.server_command(runner.InferConfig(model="models/x/model-Q4_K_M.gguf"))


def test_the_request_sends_the_image_first_then_the_same_instruction():
    payload = llamacpp_runner.chat_request(b"\x89PNG-bytes", "image/png", _cfg(max_new_tokens=1024))
    content = payload["messages"][0]["content"]
    assert [part["type"] for part in content] == ["image_url", "text"]
    assert content[1]["text"] == INSTRUCTION
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert base64.b64decode(content[0]["image_url"]["url"].split(",", 1)[1]) == b"\x89PNG-bytes"
    assert payload["temperature"] == 0.0 and payload["top_k"] == 1 and payload["max_tokens"] == 1024
    assert payload["cache_prompt"] is False  # a reused prefix broke every receipt after the fourth
    json.dumps(payload)  # must be JSON-serialisable for the HTTP request


def test_media_type_follows_the_file_and_only_length_means_truncated():
    assert llamacpp_runner.media_type("a/b/cord-test-0091.png") == "image/png"
    assert llamacpp_runner.media_type("a/b/cord-train-0569.JPG") == "image/jpeg"
    assert llamacpp_runner.is_truncated("length")
    assert not llamacpp_runner.is_truncated("stop")


def test_the_backend_registry_knows_llamacpp():
    assert backends.get("llamacpp") is llamacpp_runner


def test_the_config_accepts_every_registered_backend():
    """The config Literal, the registry and run_infer.py's --backend choices must list the same engines.

    They drifted once: --backend llamacpp parsed fine and then failed config validation on the GPU box.
    """
    names = get_args(runner.InferConfig.model_fields["backend"].annotation)
    assert set(names) == {"hf", "vllm", "llamacpp"}
    for name in names:
        assert runner.InferConfig(backend=name).backend == name
        backends.get(name)
    choices = re.search(r'"--backend",\s*choices=(\[[^\]]*\])', Path("scripts/run_infer.py").read_text())
    assert set(ast.literal_eval(choices.group(1))) == set(names)
