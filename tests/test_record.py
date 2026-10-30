import json
from dataclasses import asdict
from datetime import datetime

import pytest

from dvx.record import (
    SCHEMA_VERSION,
    RunRecord,
    append_to_corpus,
    env_info,
    new_run_id,
    read_corpus,
    save_record,
)


def make_record(run_id="20261030-210509-abcdef01"):
    return RunRecord(
        schema_version=SCHEMA_VERSION, run_id=run_id,
        created_at="2026-10-30T21:05:09", status="complete",
        engine_version="0.1.0", profile="quick", domain="general", suite_version="1",
        env={"python": "3"}, dataset={"path": "d.jsonl"}, model={"id": "m"},
        seeds=[0, 1], config={"lr": 1e-4}, results={"duplicates": {"n": 1}},
        compute={"steps": {}},
    )


def test_run_id():
    assert new_run_id(datetime(2026, 10, 30, 21, 5, 9), "abcdef0123" + "0" * 54) \
        == "20261030-210509-abcdef01"


def test_roundtrip(tmp_path):
    rec = make_record()
    path = save_record(rec, tmp_path / "run")
    assert path.name == "record.json"
    assert json.loads(path.read_text()) == asdict(rec)
    assert not (tmp_path / "run" / "record.json.tmp").exists()

    corpus = tmp_path / "sub" / "corpus.jsonl"
    assert read_corpus(corpus) == []
    append_to_corpus(rec, corpus)
    with pytest.raises(ValueError):
        append_to_corpus(rec, corpus)
    append_to_corpus(make_record("other"), corpus)
    rows = read_corpus(corpus)
    assert len(rows) == 2 and rows[0] == asdict(rec)
    assert len(corpus.read_text().splitlines()) == 2


def test_nan_rejected(tmp_path):
    rec = make_record()
    rec.results = {"x": float("nan")}
    with pytest.raises(ValueError):
        save_record(rec, tmp_path)
    assert not (tmp_path / "record.json").exists()
    with pytest.raises(ValueError):
        append_to_corpus(rec, tmp_path / "c.jsonl")
    assert read_corpus(tmp_path / "c.jsonl") == []


def test_env_info():
    assert set(env_info("cpu")) == {"python", "platform", "torch", "transformers", "peft",
                                    "datasets", "device", "dvx"}
