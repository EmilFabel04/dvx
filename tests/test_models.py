from datetime import date

import pytest
import torch

from dvx.models import REGISTRY, model_info, pick_device, release

BASE_REVISIONS = {
    "Qwen/Qwen3-0.6B-Base": "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
    "Qwen/Qwen3-1.7B-Base": "ea980cb0a6c2ae4b936e82123acc929f1cec04c1",
    "Qwen/Qwen3-4B-Base": "906bfd4b4dc7f14ee4320094d8b41684abff8539",
    "Qwen/Qwen3-8B-Base": "49e3418fbbbca6ecbdf9608b4d22e5a407081db4",
}


def test_registry_qwen3_base():
    i = model_info("Qwen/Qwen3-0.6B-Base")
    assert i.reference_id == "Qwen/Qwen3-1.7B-Base" and i.unseen_after == date(2025, 5, 1)
    assert i.revision == "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
    assert (i.family, i.params_b, i.pretrain_tokens) == ("qwen3", 0.6, 36e12)


def test_registry_holds_only_pinned_base_checkpoints():
    assert {k: v.revision for k, v in REGISTRY.items()} == BASE_REVISIONS


def test_reference_chain():
    assert model_info("Qwen/Qwen3-1.7B-Base").reference_id == "Qwen/Qwen3-4B-Base"
    assert model_info("Qwen/Qwen3-4B-Base").reference_id == "Qwen/Qwen3-8B-Base"
    assert model_info("Qwen/Qwen3-8B-Base").reference_id is None


@pytest.mark.parametrize("model_id", ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"])
def test_post_trained_checkpoints_are_unregistered(model_id):
    i = model_info(model_id)
    assert (i.family, i.revision, i.reference_id) == ("unknown", None, None)


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


def test_release_frees_parameters(tiny):
    model, _ = tiny
    release(model)
    assert all(p.is_meta for p in model.parameters())


@pytest.mark.parametrize("device", ["mps", "cuda", "cpu"])
def test_release_cached_memory_empties_that_device_cache(device, monkeypatch):
    from dvx.models import release_cached_memory

    calls = []
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: calls.append("mps"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("cuda"))
    release_cached_memory(torch.device(device))
    release_cached_memory(device)
    assert calls == ([device, device] if device != "cpu" else [])


def test_release_empties_cache_through_helper(tiny, monkeypatch):
    import dvx.models

    model, _ = tiny
    calls = []
    monkeypatch.setattr(dvx.models, "release_cached_memory", calls.append)
    release(model)
    assert [d.type for d in calls] == ["cpu"]
