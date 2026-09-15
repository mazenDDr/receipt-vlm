import os
import sys

import pytest

from receipt_vlm.train import qlora


def test_alloc_conf_is_set_before_torch_starts(monkeypatch):
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "placeholder")  # restored after the test
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    qlora.apply_cuda_alloc_conf("expandable_segments:True")
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


def test_no_alloc_conf_leaves_the_environment_alone(monkeypatch):
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    qlora.apply_cuda_alloc_conf(None)
    assert "PYTORCH_CUDA_ALLOC_CONF" not in os.environ


def test_setting_it_after_torch_is_imported_is_refused(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", object())
    with pytest.raises(RuntimeError, match="before torch"):
        qlora.apply_cuda_alloc_conf("expandable_segments:True")
