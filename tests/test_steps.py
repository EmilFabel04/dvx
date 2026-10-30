import json

import pytest

from dvx.run.steps import cached_step


def test_cached_step_runs_once(tmp_path):
    calls = []

    def fn():
        calls.append(1)
        return {"x": 1}

    first = cached_step(tmp_path, "a", fn)
    second = cached_step(tmp_path, "a", fn)
    assert len(calls) == 1
    assert first == second
    assert first["x"] == 1 and first["_wall_seconds"] >= 0.0
    assert json.loads((tmp_path / "steps" / "a.json").read_text()) == first
    assert not (tmp_path / "steps" / "a.json.tmp").exists()


def test_partial_tmp_ignored(tmp_path):
    steps = tmp_path / "steps"
    steps.mkdir()
    (steps / "x.json.tmp").write_text('{"x": ')
    out = cached_step(tmp_path, "x", lambda: {"x": 2})
    assert out["x"] == 2
    assert json.loads((steps / "x.json").read_text())["x"] == 2
    assert not (steps / "x.json.tmp").exists()


def test_failed_step_writes_nothing(tmp_path):
    def boom():
        raise RuntimeError("out of memory")

    with pytest.raises(RuntimeError):
        cached_step(tmp_path, "x", boom)
    assert not (tmp_path / "steps" / "x.json").exists()
    assert cached_step(tmp_path, "x", lambda: {"x": 3})["x"] == 3
