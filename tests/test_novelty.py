import json

import numpy as np
import pytest

from dvx.data import Example
from dvx.scan.novelty import classify, mismatched_pairs, novelty_unavailable


def test_mismatched_pairs():
    exs = [Example(f"p{i}", f"r{i}", None, i) for i in range(5)]
    pairs = mismatched_pairs(exs, 3, seed=0)
    assert len(pairs) == 3
    own = {e.prompt: e.response for e in exs}
    assert all(own[p.prompt] != p.response for p in pairs)
    with pytest.raises(ValueError):
        mismatched_pairs(exs[:1], 3, seed=0)


def test_classify():
    r = classify(
        target=[0.5, 2.0, 3.0],
        reference=[0.45, 1.0, 3.5],
        mismatched_reference=list(np.linspace(3.0, 5.0, 21)),
        known_margin=0.1,
        noise_percentile=10.0,
    )
    assert r.labels == ["known", "learnable", "noise"]
    assert r.noise_threshold == pytest.approx(3.2)
    assert r.fractions["known"] == pytest.approx(1 / 3)
    json.dumps(r.summary())
    assert "labels" not in r.summary()


def test_unavailable():
    r = novelty_unavailable("no reference model", 4)
    assert r.status == "unavailable" and r.reason == "no reference model"
    json.dumps(r.summary())


def test_mismatched_pairs_n_one():
    exs = [Example(f"p{i}", f"r{i}", None, i) for i in range(5)]
    with pytest.raises(ValueError):
        mismatched_pairs(exs, 1, seed=0)
