import dataclasses
import json

import torch
from conftest import TINY, TINY_REV

from dvx.config import TrainConfig
from dvx.data import Example
from dvx.models import count_params, load_model
from dvx.prompting import encode_sft
from dvx.scan.losses import response_losses
from dvx.train import finetune_lora


def _exs(n=16):
    return [Example("Q", "the answer is blue", None, i) for i in range(n)]


def test_loss_decreases():
    m, tok = load_model(TINY, TINY_REV, "cpu")
    exs = _exs()
    cfg = TrainConfig(lr=1e-2, epochs=3, max_seq_len=64)
    before = response_losses(m, tok, exs[:1], max_len=64)[0]
    m2, st = finetune_lora(m, tok, exs, cfg, seed=0)
    assert response_losses(m2, tok, exs[:1], max_len=64)[0] < before
    assert st.n_examples == 16 and st.steps == 6
    assert not m2.training


def test_tokens_and_flops():
    m, tok = load_model(TINY, TINY_REV, "cpu")
    exs = _exs(8)
    cfg = TrainConfig(epochs=3, max_seq_len=64)
    n_params = count_params(m)
    expected = 3 * sum(len(encode_sft(tok, e, 64).input_ids) for e in exs)
    _, st = finetune_lora(m, tok, exs, cfg, seed=0)
    assert st.tokens_trained == expected
    assert st.est_flops == 6 * n_params * expected


def test_seed_determinism():
    cfg = TrainConfig(lr=1e-2, epochs=2, max_seq_len=64)
    losses = []
    for _ in range(2):
        m, tok = load_model(TINY, TINY_REV, "cpu")
        _, st = finetune_lora(m, tok, _exs(8), cfg, seed=0)
        losses.append(st.final_loss)
    assert losses[0] == losses[1]


def test_counts_truncated_and_json_plain():
    m, tok = load_model(TINY, TINY_REV, "cpu")
    exs = _exs(3) + [Example("Q", "word " * 200, None, 99)]
    _, st = finetune_lora(m, tok, exs, TrainConfig(epochs=1, max_seq_len=32), seed=0)
    assert st.n_truncated == 1
    json.dumps(dataclasses.asdict(st))


def test_adapter_float32_with_bf16_base():
    m, tok = load_model(TINY, TINY_REV, "cpu", dtype=torch.bfloat16)
    m2, _ = finetune_lora(m, tok, _exs(2), TrainConfig(epochs=1, max_seq_len=64), seed=0)
    trainable = [p for p in m2.parameters() if p.requires_grad]
    assert trainable and all(p.dtype == torch.float32 for p in trainable)


def test_collate_pads_to_multiple_of_64_with_ignored_labels():
    from dvx.prompting import IGNORE, EncodedExample
    from dvx.train import _collate

    batch = [
        EncodedExample([5, 6, 7], [IGNORE, 6, 7], False),
        EncodedExample(list(range(10, 80)), [IGNORE] * 10 + list(range(20, 80)), False),
    ]
    ids, labels, mask = _collate(batch, 0, torch.device("cpu"), 512)
    assert ids.shape == labels.shape == mask.shape == (2, 128)
    assert ids[0, :3].tolist() == [5, 6, 7] and (ids[0, 3:] == 0).all()
    assert labels[0, :3].tolist() == [IGNORE, 6, 7] and (labels[0, 3:] == IGNORE).all()
    assert (labels[1, 70:] == IGNORE).all() and labels[1, 10:70].tolist() == list(range(20, 80))
    assert mask.sum(-1).tolist() == [3, 70] and (mask[1, 70:] == 0).all()
    assert _collate(batch, 0, torch.device("cpu"), 100)[0].shape == (2, 100)


def test_finetune_releases_cached_memory_once(monkeypatch):
    import dvx.train

    m, tok = load_model(TINY, TINY_REV, "cpu")
    calls = []
    monkeypatch.setattr(dvx.train, "release_cached_memory", calls.append)
    finetune_lora(m, tok, _exs(4), TrainConfig(epochs=1, max_seq_len=64), seed=0)
    assert [d.type for d in calls] == ["cpu"]


def _one_step(checkpointing):
    m, tok = load_model(TINY, TINY_REV, "cpu")
    use_cache = []  # what the decoder is asked for; True would warn under checkpointing
    m.get_decoder().register_forward_pre_hook(
        lambda mod, args, kwargs: use_cache.append(kwargs.get("use_cache")), with_kwargs=True
    )
    cfg = TrainConfig(lr=1e-2, max_seq_len=64, gradient_checkpointing=checkpointing)
    tuned, st = finetune_lora(m, tok, _exs(8), cfg, seed=0)  # 8 = micro 4 x accum 2
    assert st.steps == 1 and m.is_gradient_checkpointing is checkpointing
    assert use_cache == [False, False]  # one forward per micro-batch, KV cache off
    lora = {n: p.detach().clone() for n, p in tuned.named_parameters() if p.requires_grad}
    return st.final_loss, lora


def test_gradient_checkpointing_matches_plain_training():
    # Default lora_dropout (0.05): torch.utils.checkpoint preserves RNG state on
    # recompute by default (preserve_rng_state=True), so dropout masks match.
    loss_on, lora_on = _one_step(True)
    loss_off, lora_off = _one_step(False)
    assert abs(loss_on - loss_off) < 1e-6
    assert lora_on.keys() == lora_off.keys() and lora_on
    assert all(torch.allclose(lora_on[n], lora_off[n], atol=1e-6) for n in lora_on)
