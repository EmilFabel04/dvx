import json

import pytest

from dvx.data import (
    DatasetFormatError,
    Example,
    fingerprint,
    load_control_examples,
    load_examples,
    sample_examples,
)

GOOD = json.dumps({"prompt": "p", "response": "r"})
MULTI_TURN_LINE = json.dumps(
    {
        "messages": [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
        ]
    }
)


def _write(tmp_path, text):
    p = tmp_path / "d.jsonl"
    p.write_text(text)
    return p


def test_prompt_response(tmp_path):
    p = _write(tmp_path, GOOD + "\n" + json.dumps({"prompt": "p2", "response": "r2"}) + "\n")
    assert load_examples(p) == [
        Example(prompt="p", response="r", system=None, line=1),
        Example(prompt="p2", response="r2", system=None, line=2),
    ]


def test_messages_with_system(tmp_path):
    line = json.dumps(
        {
            "messages": [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "u"},
                {"role": "assistant", "content": "a"},
            ]
        }
    )
    p = _write(tmp_path, line + "\n")
    assert load_examples(p) == [Example(prompt="u", response="a", system="s", line=1)]


def test_blank_lines_skipped(tmp_path):
    p = _write(tmp_path, "\n" + GOOD + "\n")
    assert [e.line for e in load_examples(p)] == [2]


@pytest.mark.parametrize(
    "bad, msg",
    [
        ("{not json", "invalid JSON"),
        ('{"prompt": "p"}', "missing"),
        (MULTI_TURN_LINE, "multi-turn"),
        ('{"prompt": 1, "response": "r"}', "must be strings"),
        ('{"prompt": "p", "response": "  "}', "empty response"),
    ],
)
def test_errors_name_line(tmp_path, bad, msg):
    p = _write(tmp_path, GOOD + "\n" + bad + "\n")
    with pytest.raises(DatasetFormatError) as e:
        load_examples(p)
    assert "line 2" in str(e.value) and msg in str(e.value)


def test_empty_file(tmp_path):
    with pytest.raises(DatasetFormatError, match="no examples"):
        load_examples(_write(tmp_path, "\n\n"))


def _exs(n):
    return [Example(prompt=f"p{i}", response=f"r{i}", system=None, line=i + 1) for i in range(n)]


def test_fingerprint():
    exs = _exs(3)
    fp = fingerprint(exs)
    assert len(fp) == 64 and int(fp, 16) >= 0
    assert fp == fingerprint(_exs(3))
    assert fp != fingerprint(list(reversed(exs)))


def test_sample_under_cap():
    exs = _exs(5)
    assert sample_examples(exs, 10, 0) == exs


def test_sample_over_cap():
    exs = _exs(100)
    a = sample_examples(exs, 10, 0)
    assert len(a) == 10
    assert a == sample_examples(exs, 10, 0)
    lines = [e.line for e in a]
    assert lines == sorted(lines)
    assert a != sample_examples(exs, 10, 1)


def test_control_from_rows():
    rows = [
        {"instruction": "i", "context": "", "response": "r", "category": "x"},
        {"instruction": "i2", "context": "c2", "response": "r2", "category": "y"},
    ]
    assert {e.prompt for e in load_control_examples(2, 0, rows)} == {"i", "i2\n\nc2"}
    with pytest.raises(ValueError):
        load_control_examples(3, 0, rows)
