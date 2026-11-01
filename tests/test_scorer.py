from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from dvx.evals.registry import MCQItem
from dvx.evals.scorer import letter_token_ids, score_mcq
from dvx.prompting import encode_prompt_text, render_mcq

ITEM = MCQItem("b", "1", "Q?", ("x", "y"), 1)
SHOTS = (
    MCQItem("b", "s1", "S1?", ("p", "q", "r"), 2),
    MCQItem("b", "s2", "S2?", ("u", "v"), 0),
)


def test_render_mcq_block():
    assert render_mcq(ITEM) == "Question: Q?\nA. x\nB. y\nAnswer:"


def test_render_mcq_with_shots():
    assert render_mcq(ITEM, SHOTS) == (
        "Question: S1?\nA. p\nB. q\nC. r\nAnswer: C\n\n"
        "Question: S2?\nA. u\nB. v\nAnswer: A\n\n"
        "Question: Q?\nA. x\nB. y\nAnswer:"
    )


def test_letter_ids_are_the_space_prefixed_letters_after_answer(tiny):
    _, tok = tiny
    ids = letter_token_ids(tok, 4)
    assert len(set(ids)) == 4
    for letter, i in zip("ABCD", ids, strict=True):
        assert tok(f"Answer: {letter}", add_special_tokens=False).input_ids[-1] == i


def test_letter_ids_reject_a_letter_that_is_not_one_token():
    class TwoTokenLetters:
        def __call__(self, text, add_special_tokens=False):
            ids = [ord(c) for c in text]
            return SimpleNamespace(input_ids=ids)

    with pytest.raises(ValueError, match="one token"):
        letter_token_ids(TwoTokenLetters(), 2)


class StubLM(nn.Module):
    """Favors letter B at the last position only; earlier positions favor A."""

    def __init__(self, tok):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1))
        self.a, self.b = letter_token_ids(tok, 2)
        self.vocab = len(tok)
        self.seen: list[list[int]] = []

    def forward(self, input_ids, attention_mask=None, **kwargs):
        assert kwargs.get("logits_to_keep") == 1
        assert kwargs.get("use_cache") is False  # no KV cache for one-shot scoring
        for row, mask in zip(input_ids.tolist(), attention_mask.tolist(), strict=True):
            self.seen.append([t for t, m in zip(row, mask, strict=True) if m])
        logits = torch.zeros(*input_ids.shape, self.vocab)
        logits[:, :, self.a] = 5.0
        logits[:, -1, self.a] = 0.0
        logits[:, -1, self.b] = 5.0
        return SimpleNamespace(logits=logits[:, -1:])


def _items(n):
    return [MCQItem("b", str(i), "word " * (i * 7) + "Q?", ("x", "y"), i % 2) for i in range(n)]


def test_scoring_uses_last_logits(tiny):
    _, tok = tiny
    got = score_mcq(StubLM(tok), tok, _items(5), batch_size=2)
    assert got.correct.tolist() == [False, True, False, True, False]
    assert got.predicted.tolist() == [1] * 5
    assert got.n_truncated == 0


def test_scoring_prompts_carry_the_shots(tiny):
    _, tok = tiny
    model = StubLM(tok)
    score_mcq(model, tok, [ITEM], shots=SHOTS, batch_size=2)
    assert model.seen == [encode_prompt_text(tok, render_mcq(ITEM, SHOTS))]


def test_left_truncation_is_counted_and_keeps_the_tail(tiny):
    _, tok = tiny
    model = StubLM(tok)
    items = _items(4)  # prompts grow by 7 words per item
    lengths = [len(encode_prompt_text(tok, render_mcq(it, SHOTS))) for it in items]
    cap = lengths[1]
    got = score_mcq(model, tok, items, shots=SHOTS, batch_size=4, max_len=cap)
    assert got.n_truncated == sum(n > cap for n in lengths) == 2
    for it, seen in zip(items, model.seen, strict=True):  # one batch, shortest first
        assert seen == encode_prompt_text(tok, render_mcq(it, SHOTS))[-cap:]


def test_real_tiny_deterministic(tiny):
    model, tok = tiny
    model.eval()
    items = [
        MCQItem("b", "1", "Q?", ("x", "y", "z"), 2),
        MCQItem("b", "2", "A longer question here?", ("p", "q", "r", "s", "t"), 0),
        MCQItem("b", "3", "Hi", ("m", "n"), 1),
    ]
    a = score_mcq(model, tok, items, shots=SHOTS, batch_size=2, max_len=64)
    b = score_mcq(model, tok, items, shots=SHOTS, batch_size=2, max_len=64)
    assert a.correct.dtype == bool and len(a.correct) == 3
    assert np.array_equal(a.correct, b.correct) and np.array_equal(a.predicted, b.predicted)
    assert all(0 <= p < len(it.options) for p, it in zip(a.predicted, items, strict=True))


def test_score_mcq_releases_cached_memory_once(tiny, monkeypatch):
    import dvx.evals.scorer

    _, tok = tiny
    calls = []
    monkeypatch.setattr(dvx.evals.scorer, "release_cached_memory", calls.append)
    items = [MCQItem("b", str(i), "Q?", ("x", "y"), 0) for i in range(3)]
    score_mcq(StubLM(tok), tok, items, batch_size=2)
    assert [d.type for d in calls] == ["cpu"]
