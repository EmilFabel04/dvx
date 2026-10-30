from datetime import date

import torch

from dvx.models import model_info, pick_device


def test_registry_qwen3():
    i = model_info("Qwen/Qwen3-0.6B")
    assert i.reference_id == "Qwen/Qwen3-1.7B" and i.unseen_after == date(2025, 5, 1)
    assert i.revision == "c1899de289a04d12100db370d81485cdf75e47ca"


def test_reference_chain():
    assert model_info("Qwen/Qwen3-1.7B").reference_id == "Qwen/Qwen3-4B"
    assert model_info("Qwen/Qwen3-4B").reference_id == "Qwen/Qwen3-8B"
    assert model_info("Qwen/Qwen3-8B").reference_id is None


def test_unknown_model():
    i = model_info("nobody/nothing")
    assert i.family == "unknown"
    assert i.reference_id is None and i.unseen_after is None and i.revision is None


def test_pick_device():
    assert pick_device() in {"cuda", "mps", "cpu"}


def test_load_tiny(tiny):
    model, tok = tiny
    assert tok.pad_token_id is not None
    assert model.dtype == torch.float32
    assert not model.training
