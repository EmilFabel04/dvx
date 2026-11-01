import numpy as np

from dvx.data import Example
from dvx.scan.losses import response_losses, token_logprobs


def _exs():
    texts = ["hi", "a much longer question about many things here", "mid length one", "x"]
    return [
        Example(prompt=t, response=t[::-1] + " answer", system=None, line=i + 1)
        for i, t in enumerate(texts)
    ]


def test_losses_positive(tiny):
    model, tok = tiny
    out = response_losses(model, tok, _exs())
    assert len(out) == 4 and all(isinstance(x, float) and x > 0 for x in out)


def test_losses_batch_invariant(tiny):
    model, tok = tiny
    a = response_losses(model, tok, _exs(), batch_size=1)
    b = response_losses(model, tok, _exs(), batch_size=3)
    assert np.allclose(a, b, rtol=1e-4)


def test_token_logprobs(tiny):
    model, tok = tiny
    texts = ["hello world", "a somewhat longer piece of text"]
    out = token_logprobs(model, tok, texts, batch_size=2)
    for t, lp in zip(texts, out, strict=True):
        assert len(lp) == len(tok(t).input_ids) - 1 and (lp <= 0).all()


def test_losses_match_hf_loss(tiny):
    import torch

    from dvx.prompting import encode_sft

    model, tok = tiny
    ex = _exs()[1]
    enc = encode_sft(tok, ex, 512)
    ids = torch.tensor([enc.input_ids])
    labels = torch.tensor([enc.labels])
    with torch.no_grad():
        expected = model(input_ids=ids, labels=labels).loss.item()
    got = response_losses(model, tok, [ex], batch_size=1)[0]
    assert np.isclose(got, expected, rtol=1e-4)


def test_token_logprobs_match_reference(tiny):
    import torch

    model, tok = tiny
    text = "hello world, this is a test"
    ids = torch.tensor([tok(text).input_ids])
    with torch.no_grad():
        lp = torch.log_softmax(model(input_ids=ids).logits.float(), -1)
    expected = lp[0, :-1].gather(-1, ids[0, 1:, None])[:, 0].numpy()
    got = token_logprobs(model, tok, [text])[0]
    assert np.allclose(got, expected, rtol=1e-4, atol=1e-5)


def _reference_logprobs(model, ids, mask):
    import torch

    with torch.no_grad():
        lp = torch.log_softmax(model(input_ids=ids, attention_mask=mask).logits.float(), -1)
    return lp[:, :-1].gather(-1, ids[:, 1:, None])[..., 0]


def _padded_batch(tok, lengths):
    import torch

    gen = torch.Generator().manual_seed(0)
    width = max(lengths)
    ids = torch.full((len(lengths), width), tok.pad_token_id)
    mask = torch.zeros(len(lengths), width, dtype=torch.long)
    for row, n in enumerate(lengths):
        ids[row, :n] = torch.randint(3, len(tok), (n,), generator=gen)
        mask[row, :n] = 1
    return ids, mask


def test_target_logprobs_chunked_match_full_logits(tiny):
    import torch

    from dvx.scan import losses

    model, tok = tiny
    ids, mask = _padded_batch(tok, [2 * losses._CHUNK + 6, 9])  # 3 chunks of targets
    widths = []
    head = model.get_output_embeddings()
    hook = head.register_forward_hook(lambda mod, inp, out: widths.append(out.shape[1]))
    try:
        with torch.no_grad():
            got = losses._target_logprobs(model, ids, mask)
    finally:
        hook.remove()
    assert got.dtype == torch.float32 and got.shape == (2, ids.shape[1] - 1)
    assert len(widths) == 3 and max(widths) <= losses._CHUNK  # never the full sequence
    assert torch.allclose(got, _reference_logprobs(model, ids, mask), atol=1e-5)


def test_target_logprobs_falls_back_without_output_embeddings(tiny, monkeypatch):
    import torch

    from dvx.scan import losses

    model, tok = tiny
    ids, mask = _padded_batch(tok, [20, 7])
    monkeypatch.setattr(model, "get_output_embeddings", lambda: None)
    with torch.no_grad():
        got = losses._target_logprobs(model, ids, mask)
    assert torch.allclose(got, _reference_logprobs(model, ids, mask), atol=1e-5)


def test_padded_width_buckets_to_64_capped_at_max_len():
    from dvx.prompting import padded_width

    assert [padded_width(n, 512) for n in (1, 63, 64, 65, 130)] == [64, 64, 64, 128, 192]
    assert padded_width(500, 512) == 512 and padded_width(512, 512) == 512
    assert padded_width(70, 100) == 100 and padded_width(20, 32) == 32


def test_scoring_batches_use_bucketed_widths(tiny, monkeypatch):
    from dvx.scan import losses

    model, tok = tiny
    widths = []
    real = losses._target_logprobs

    def spy(m, ids, mask):
        widths.append(ids.shape[1])
        return real(m, ids, mask)

    monkeypatch.setattr(losses, "_target_logprobs", spy)
    response_losses(model, tok, _exs(), max_len=512, batch_size=2)
    token_logprobs(model, tok, ["hello world", "word " * 80], max_len=70, batch_size=1)
    assert widths[:2] == [64, 64] and widths[2:] == [64, 70]


def test_release_cached_memory_once_per_scoring_call(tiny, monkeypatch):
    from dvx.scan import losses

    model, tok = tiny
    calls = []
    monkeypatch.setattr(losses, "release_cached_memory", calls.append)
    response_losses(model, tok, _exs(), batch_size=2)
    token_logprobs(model, tok, ["hello world", "another text"], batch_size=1)
    assert [d.type for d in calls] == ["cpu", "cpu"]


def test_target_logprobs_falls_back_for_unlisted_model_type(tiny, monkeypatch):
    import torch

    from dvx.scan import losses

    model, tok = tiny
    ids, mask = _padded_batch(tok, [2 * losses._CHUNK + 6, 9])
    monkeypatch.setattr(model.config, "model_type", "gemma2")
    widths = []
    hook = model.get_output_embeddings().register_forward_hook(
        lambda mod, inp, out: widths.append(out.shape[1])
    )
    try:
        with torch.no_grad():
            got = losses._target_logprobs(model, ids, mask)
    finally:
        hook.remove()
    assert widths == [ids.shape[1]]  # one full-sequence head call: the fallback
    assert torch.allclose(got, _reference_logprobs(model, ids, mask), atol=1e-5)
