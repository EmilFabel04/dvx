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
