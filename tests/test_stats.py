import dataclasses
import json

import numpy as np
import pytest

from dvx.evals.stats import (
    Estimate,
    baseline_check,
    chance_level,
    group_accuracy,
    paired_delta,
)


def test_group_accuracy():
    e = group_accuracy({"a": np.array([1, 1, 0, 0], bool)}, 200, 0)
    assert e.value == 50.0 and e.ci_low <= 50.0 <= e.ci_high


def test_paired_delta_macro():
    a = {"x": np.zeros(4, bool), "y": np.ones(4, bool)}
    b = {"x": np.array([1, 1, 0, 0], bool), "y": np.ones(4, bool)}
    assert paired_delta(a, b, 200, 0).value == 25.0


def test_identical_zero():
    a = {"x": np.array([1, 0, 1, 1, 0], bool), "y": np.array([1, 1, 0], bool)}
    e = paired_delta(a, {k: v.copy() for k, v in a.items()}, 200, 0)
    assert (e.value, e.ci_low, e.ci_high) == (0.0, 0.0, 0.0)


def test_ci_ordered_and_deterministic():
    rng = np.random.default_rng(1)
    a = {"x": rng.random(500) < 0.4, "y": rng.random(144) < 0.5, "z": rng.random(100) < 0.3}
    b = {k: v | (rng.random(len(v)) < 0.1) for k, v in a.items()}
    for e in (group_accuracy(a, 300, 0), paired_delta(a, b, 300, 0)):
        assert e.ci_low <= e.value <= e.ci_high
    assert paired_delta(a, b, 300, 7) == paired_delta(a, b, 300, 7)
    assert paired_delta(a, b, 300, 0).value > 0


def test_paired_delta_validates():
    with pytest.raises(ValueError):
        paired_delta({"x": np.ones(3, bool)}, {"y": np.ones(3, bool)}, 10, 0)
    with pytest.raises(ValueError):
        paired_delta({"x": np.ones(3, bool)}, {"x": np.ones(4, bool)}, 10, 0)


def test_json_safe():
    a = {"x": np.array([1, 0, 1, 1], bool)}
    e = group_accuracy(a, 50, 0)
    c = baseline_check(a, {"x": 0.25}, 0.05)
    json.dumps(dataclasses.asdict(e))
    json.dumps(dataclasses.asdict(c))
    assert type(e.value) is float and type(c.ok) is bool


def test_baseline():
    correct = {"a": np.array([1] * 3 + [0] * 7, bool), "b": np.array([1] * 4 + [0] * 6, bool)}
    chance = {"a": 0.25, "b": 1 / 3}
    ok = baseline_check(correct, chance, 0.05)
    assert ok.ok and ok.margin_pp == pytest.approx(5.8333, abs=1e-3)
    assert ok.required_pp == pytest.approx(5.0)
    assert ok.per_benchmark_pp["a"] == pytest.approx(5.0)
    assert ok.chance_pp == pytest.approx({"a": 25.0, "b": 100 / 3})
    assert not baseline_check(correct, chance, 0.10).ok


def test_chance_level_is_the_larger_of_uniform_and_majority():
    assert chance_level([0, 1, 2, 3], 4) == 0.25
    assert chance_level([0, 0, 0, 1], 4) == 0.75
    assert chance_level([0, 1, 1], 3) == pytest.approx(2 / 3)
    assert chance_level([0, 1, 2, 2], 3) == 0.5


def test_always_a_on_skewed_answers_fails_the_gate():
    # 45% of answers are A: a model that always answers A has no skill, yet beat 1/k by 20 pp.
    answers = [0] * 45 + [1] * 20 + [2] * 20 + [3] * 15
    always_a = {"medqa": np.array([a == 0 for a in answers])}
    check = baseline_check(always_a, {"medqa": chance_level(answers, 4)}, 0.05)
    assert not check.ok and check.per_benchmark_pp["medqa"] == pytest.approx(0.0)


def test_just_above_uniform_but_below_majority_is_below_chance():
    answers = [0] * 30 + [1] * 25 + [2] * 25 + [3] * 20  # majority 30% > 1/k 25%
    correct = {"medqa": np.array([True] * 27 + [False] * 73)}
    check = baseline_check(correct, {"medqa": chance_level(answers, 4)}, 0.05)
    assert check.per_benchmark_pp["medqa"] == pytest.approx(-3.0)


def test_includes_zero():
    assert Estimate(1.0, -0.5, 2.0).includes_zero() and not Estimate(1.0, 0.2, 2.0).includes_zero()
