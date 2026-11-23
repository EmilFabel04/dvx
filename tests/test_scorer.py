from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from dvx.evals.registry import MCQItem
from dvx.evals.scorer import letter_token_ids, score_mcq
from dvx.prompting import render_mcq


def test_render_mcq_plain(tiny):
    _, tok = tiny
    tok.chat_template = None
    t = render_mcq(tok, MCQItem("b", "1", "Q?", ("x", "y"), 1))
    assert "A. x\nB. y" in t and t.endswith("Answer:")


def test_render_mcq_chat(tiny):
    _, tok = tiny
    tok.chat_template = (
        "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}\n{% endfor %}"
        "{% if add_generation_prompt %}<|assistant|>{% endif %}"
    )
    t = render_mcq(tok, MCQItem("b", "1", "Q?", ("x", "y"), 1))
    assert t.startswith("<|user|>Q?\n\nA. x\nB. y\n\nAnswer with the letter") and t.endswith(
        "<|assistant|>"
    )


def test_letter_ids(tiny):
    _, tok = tiny
    ids = letter_token_ids(tok, 4)
    assert len(ids) == 4 and all(ids)
    flat = [i for g in ids for i in g]
    assert len(flat) == len(set(flat))


class StubLM(nn.Module):
    """Favors letter B at the last position only; earlier positions favor A."""

    def __init__(self, tok):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1))
        self.a, self.b = letter_token_ids(tok, 2)
        self.vocab = len(tok)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        logits = torch.zeros(*input_ids.shape, self.vocab)
        logits[:, :, self.a] = 5.0
        logits[:, -1, self.a] = 0.0
        logits[:, -1, self.b] = 5.0
        assert kwargs.get("logits_to_keep") == 1
        return SimpleNamespace(logits=logits[:, -1:])


def test_scoring_uses_last_logits(tiny):
    _, tok = tiny
    items = [MCQItem("b", str(i), "word " * (i * 7) + "Q?", ("x", "y"), i % 2) for i in range(5)]
    got = score_mcq(StubLM(tok), tok, items, batch_size=2)
    assert got.tolist() == [False, True, False, True, False]


def test_real_tiny_deterministic(tiny):
    model, tok = tiny
    model.eval()
    items = [
        MCQItem("b", "1", "Q?", ("x", "y", "z"), 2),
        MCQItem("b", "2", "A longer question here?", ("p", "q", "r", "s", "t"), 0),
        MCQItem("b", "3", "Hi", ("m", "n"), 1),
    ]
    a = score_mcq(model, tok, items, batch_size=2, max_len=32)
    b = score_mcq(model, tok, items, batch_size=2, max_len=32)
    assert a.dtype == bool and len(a) == 3
    assert np.array_equal(a, b)
