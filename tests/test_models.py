from datetime import date

import pytest
import torch

from dvx.models import REGISTRY, model_info, pick_device, release


def test_registry_qwen3():
    i = model_info("Qwen/Qwen3-0.6B")
    assert i.reference_id == "Qwen/Qwen3-1.7B" and i.unseen_after == date(2025, 5, 1)
    assert i.revision == "c1899de289a04d12100db370d81485cdf75e47ca"


def test_registry_revisions():
    assert {k: v.revision for k, v in REGISTRY.items()} == {
        "Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
        "Qwen/Qwen3-1.7B": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "Qwen/Qwen3-4B": "1cfa9a7208912126459214e8b04321603b3df60c",
        "Qwen/Qwen3-8B": "b968826d9c46dd6066d109eabc6255188de91218",
    }


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
