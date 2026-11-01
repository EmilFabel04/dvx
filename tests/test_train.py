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


def _sft_batch(tok, lengths, n_prompt):
    """Right-padded ids, mask and labels; the first n_prompt tokens of each row are IGNORE."""
    from dvx.prompting import IGNORE

    gen = torch.Generator().manual_seed(0)
    width = max(lengths)
    ids = torch.full((len(lengths), width), tok.pad_token_id)
    labels = torch.full((len(lengths), width), IGNORE)
    mask = torch.zeros(len(lengths), width, dtype=torch.long)
    for row, n in enumerate(lengths):
        ids[row, :n] = torch.randint(3, len(tok), (n,), generator=gen)
        labels[row, n_prompt:n] = ids[row, n_prompt:n]
        mask[row, :n] = 1
    return ids, mask, labels


def _loss_and_grads(model, loss_fn):
    model.zero_grad()
    loss = loss_fn()
    loss.backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    return loss.detach(), grads


def test_sft_loss_matches_hf_loss_and_grads_in_chunks():
    from dvx import train

    m, tok = load_model(TINY, TINY_REV, "cpu")
    ids, mask, labels = _sft_batch(tok, [2 * train._LOSS_CHUNK + 6, 9], n_prompt=5)
    widths = []  # every head call, forward and backward recompute
    m.lm_head.register_forward_hook(lambda mod, inp, out: widths.append(out.shape[1]))
    got, got_grads = _loss_and_grads(m, lambda: train._sft_loss(m, ids, mask, labels))
    assert widths and max(widths) <= train._LOSS_CHUNK  # never [B, T, V] logits
    widths.clear()
    ref, ref_grads = _loss_and_grads(
        m, lambda: m(input_ids=ids, attention_mask=mask, labels=labels).loss
    )
    assert widths == [ids.shape[1]]
    assert got.dtype == torch.float32 and torch.allclose(got, ref, rtol=1e-6, atol=0)
    assert got_grads.keys() == ref_grads.keys() and got_grads
    assert all(torch.allclose(got_grads[n], ref_grads[n], rtol=1e-4, atol=1e-7) for n in ref_grads)


def test_sft_loss_falls_back_to_hf_loss_for_unlisted_model_type(monkeypatch):
    from dvx import train

    m, tok = load_model(TINY, TINY_REV, "cpu")
    ids, mask, labels = _sft_batch(tok, [2 * train._LOSS_CHUNK + 6, 9], n_prompt=5)
    monkeypatch.setattr(m.config, "model_type", "gemma2")
    widths = []
    m.lm_head.register_forward_hook(lambda mod, inp, out: widths.append(out.shape[1]))
    got = train._sft_loss(m, ids, mask, labels)
    assert widths == [ids.shape[1]]  # one full-sequence head call: HF's own loss
    ref = m(input_ids=ids, attention_mask=mask, labels=labels).loss
    assert torch.allclose(got, ref, rtol=1e-6, atol=0)


def test_finetune_never_projects_full_sequence():
    from dvx import train

    m, tok = load_model(TINY, TINY_REV, "cpu")
    widths = []
    m.lm_head.register_forward_hook(lambda mod, inp, out: widths.append(out.shape[1]))
    exs = [Example("Q", "word " * 60, None, i) for i in range(8)]  # longer than a chunk
    finetune_lora(m, tok, exs, TrainConfig(epochs=1, max_seq_len=128), seed=0)
    assert widths and max(widths) <= train._LOSS_CHUNK


def test_finetune_frees_optimizer_and_grads_before_releasing_cache(monkeypatch):
    import weakref

    import dvx.train

    optimizers = []

    class TrackedAdamW(torch.optim.AdamW):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            optimizers.append(weakref.ref(self))

    m, tok = load_model(TINY, TINY_REV, "cpu")
    at_release = []

    def release(device):
        no_grads = all(p.grad is None for p in m.parameters())
        at_release.append(([r() is None for r in optimizers], no_grads))

    monkeypatch.setattr(torch.optim, "AdamW", TrackedAdamW)
    monkeypatch.setattr(dvx.train, "release_cached_memory", release)
    tuned, _ = finetune_lora(m, tok, _exs(8), TrainConfig(max_seq_len=64), seed=0)
    assert at_release == [([True], True)]  # optimizer gone, no grads, when the cache empties
    assert all(p.grad is None for p in tuned.parameters())


def test_sft_loss_matches_peft_loss_with_gradient_checkpointing():
    from peft import LoraConfig, get_peft_model

    from dvx import train

    m, tok = load_model(TINY, TINY_REV, "cpu")
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    cfg = TrainConfig()
    peft_model = get_peft_model(
        m,
        LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=list(cfg.lora_target_modules),
            task_type="CAUSAL_LM",
        ),
    )
    peft_model.train()
    assert m.is_gradient_checkpointing
    ids, mask, labels = _sft_batch(tok, [2 * train._LOSS_CHUNK + 6, 9], n_prompt=5)

    def seeded(loss_fn):
        torch.manual_seed(0)  # the same LoRA dropout masks for both losses
        return _loss_and_grads(peft_model, loss_fn)

    got, got_grads = seeded(lambda: train._sft_loss(peft_model, ids, mask, labels))
    ref, ref_grads = seeded(
        lambda: peft_model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False).loss
    )
    assert torch.allclose(got, ref, rtol=1e-6, atol=0)
    assert got_grads.keys() == ref_grads.keys() and all("lora_" in n for n in got_grads)
    assert all(torch.allclose(got_grads[n], ref_grads[n], rtol=1e-4, atol=1e-7) for n in ref_grads)
