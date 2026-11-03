import pytest

from dvx.data import Example
from dvx.prompting import encode_sft, render_prompt


def _ex(prompt, response):
    return Example(prompt=prompt, response=response, system=None, line=1)


def _n_prompt(tok, prompt):
    """Prompt length in tokens, including the BOS the plain path prepends."""
    return len(tok(render_prompt(tok, prompt), add_special_tokens=True).input_ids)


def test_render_plain(tiny):
    _, tok = tiny
    tok.chat_template = None
    assert render_prompt(tok, "What?") == "### Question:\nWhat?\n\n### Answer:\n"
    assert render_prompt(tok, "What?", system="S").startswith("S\n\n")


def test_render_chat(tiny):
    _, tok = tiny
    tok.chat_template = (
        "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}\n{% endfor %}"
        "{% if add_generation_prompt %}<|assistant|>{% endif %}"
    )
    assert render_prompt(tok, "Q") == "<|user|>Q\n<|assistant|>"


def test_encode_masks_prompt(tiny):
    _, tok = tiny
    tok.chat_template = None
    enc = encode_sft(tok, _ex("What is it?", "It is a cat."), 128)
    n = _n_prompt(tok, "What is it?")
    assert enc.labels[:n] == [-100] * n
    assert enc.labels[n:] == enc.input_ids[n:]
    assert enc.input_ids[-1] == tok.eos_token_id
    assert len(enc.input_ids) == len(enc.labels)
    assert not enc.truncated


def test_encode_truncates_long_prompt(tiny):
    _, tok = tiny
    tok.chat_template = None
    enc = encode_sft(tok, _ex("word " * 2000, "short answer"), 64)
    resp = tok("short answer", add_special_tokens=False).input_ids + [tok.eos_token_id]
    assert len(enc.input_ids) <= 64
    assert enc.truncated
    assert [x for x in enc.labels if x != -100] == resp


def test_encode_truncates_long_response(tiny):
    _, tok = tiny
    tok.chat_template = None
    enc = encode_sft(tok, _ex("Q?", "word " * 2000), 64)
    labeled = [x for x in enc.labels if x != -100]
    assert len(labeled) <= 48 and enc.truncated
    assert enc.input_ids[-1] == tok.eos_token_id
    assert enc.labels[-1] == tok.eos_token_id
    assert enc.labels.count(-100) == _n_prompt(tok, "Q?")


def test_encode_plain_starts_with_bos(tiny):
    _, tok = tiny
    tok.chat_template = None
    enc = encode_sft(tok, _ex("Q?", "A."), 128)
    assert enc.input_ids[0] == tok.bos_token_id
    assert enc.input_ids[1] != tok.bos_token_id


def test_encode_rejects_tiny_max_len(tiny):
    _, tok = tiny
    with pytest.raises(ValueError):
        encode_sft(tok, _ex("Q?", "A."), 7)
