import pytest

from receipt_vlm.infer import backends, runner, vllm_runner


def test_the_module_imports_without_vllm_installed():
    # vLLM lives in its own env; the runner must still import here, in "main" and in "quant"
    assert callable(vllm_runner.load) and callable(vllm_runner.predict)


def test_only_a_length_stop_counts_as_truncated():
    assert vllm_runner.is_truncated("length")
    assert not vllm_runner.is_truncated("stop")
    assert not vllm_runner.is_truncated(None)


def test_the_default_backend_is_transformers_and_vllm_is_opt_in():
    assert runner.InferConfig().backend == "hf"
    assert backends.get("hf") is runner
    assert backends.get("vllm") is vllm_runner
    with pytest.raises(ValueError, match="unknown backend"):
        backends.get("tensorrt")
