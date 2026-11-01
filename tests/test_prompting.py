import json

import pytest

from dvx.data import Example, load_examples
from dvx.prompting import IGNORE, encode_sft, render_prompt

CHAT_TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}\n{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)


def _ex(prompt, response, system=None):
    return Example(prompt=prompt, response=response, system=system, line=1)


def _n_prompt(tok, prompt, system=None):
    """Prompt length in tokens, including the BOS the tokenizer prepends."""
    return len(tok(render_prompt(prompt, system), add_special_tokens=True).input_ids)


def test_render_prompt_is_plain_text():
    assert render_prompt("What?") == "What?\n\n"
    assert render_prompt("What?", system="S") == "S\n\nWhat?\n\n"


def _expected_ids(tok, prompt_text, response):
    prompt = [tok.bos_token_id, *tok(prompt_text, add_special_tokens=False).input_ids]
    return prompt, [*tok(response, add_special_tokens=False).input_ids, tok.eos_token_id]


@pytest.mark.parametrize(
    "row",
    [
        {"system": "Be brief.", "prompt": "What is it?", "response": "A cat."},
        {
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "What is it?"},
                {"role": "assistant", "content": "A cat."},
            ]
        },
    ],
    ids=["prompt_response", "messages"],
)
def test_both_input_formats_encode_as_plain_text(tiny, tmp_path, row):
    _, tok = tiny
    path = tmp_path / "d.jsonl"
    path.write_text(json.dumps(row) + "\n")
    (ex,) = load_examples(path)
    enc = encode_sft(tok, ex, 128)
    prompt, response = _expected_ids(tok, "Be brief.\n\nWhat is it?\n\n", "A cat.")
    assert enc.input_ids == prompt + response
    assert enc.labels == [IGNORE] * len(prompt) + response  # loss on response + EOS only


def test_chat_template_is_never_used(tiny):
    _, tok = tiny
    ex = _ex("What is it?", "A cat.", system="S")
    tok.chat_template = CHAT_TEMPLATE
    with_template = encode_sft(tok, ex, 128)
    tok.chat_template = None
    assert with_template == encode_sft(tok, ex, 128)


def test_encode_masks_prompt(tiny):
    _, tok = tiny
    enc = encode_sft(tok, _ex("What is it?", "It is a cat."), 128)
    n = _n_prompt(tok, "What is it?")
    assert enc.labels[:n] == [-100] * n
    assert enc.labels[n:] == enc.input_ids[n:]
    assert enc.input_ids[-1] == tok.eos_token_id
    assert len(enc.input_ids) == len(enc.labels)
    assert not enc.truncated


def test_encode_truncates_long_prompt(tiny):
    _, tok = tiny
    enc = encode_sft(tok, _ex("word " * 2000, "short answer"), 64)
    resp = tok("short answer", add_special_tokens=False).input_ids + [tok.eos_token_id]
    assert len(enc.input_ids) <= 64
    assert enc.truncated
    assert [x for x in enc.labels if x != -100] == resp


def test_encode_truncates_long_response(tiny):
    _, tok = tiny
    enc = encode_sft(tok, _ex("Q?", "word " * 2000), 64)
    labeled = [x for x in enc.labels if x != -100]
    assert len(labeled) <= 48 and enc.truncated
    assert enc.input_ids[-1] == tok.eos_token_id
    assert enc.labels[-1] == tok.eos_token_id
    assert enc.labels.count(-100) == _n_prompt(tok, "Q?")


def test_encode_starts_with_bos_when_the_tokenizer_adds_one(tiny):
    _, tok = tiny
    enc = encode_sft(tok, _ex("Q?", "A."), 128)
    assert enc.input_ids[0] == tok.bos_token_id
    assert enc.input_ids[1] != tok.bos_token_id


def test_encode_rejects_tiny_max_len(tiny):
    _, tok = tiny
    with pytest.raises(ValueError):
        encode_sft(tok, _ex("Q?", "A."), 7)
