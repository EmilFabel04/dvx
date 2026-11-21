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
