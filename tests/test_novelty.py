import json

import numpy as np
import pytest
from fixtures import medmcqa_sample

from dvx.data import Example
from dvx.models import REGISTRY
from dvx.scan.novelty import (
    bucket_examples,
    classify,
    mismatched_pairs,
    novelty_unavailable,
)

MISMATCHED = list(np.linspace(3.0, 5.0, 21))  # 10th percentile: 3.2
TRUE_PAIRS = [1.0] * 21  # every true pair scores lower than its mismatched pair


def _classify(target, reference, mismatched=MISMATCHED, true_pairs=TRUE_PAIRS):
    return classify(
        target=target,
        reference=reference,
        mismatched_reference=mismatched,
        mismatched_true_reference=true_pairs,
        known_margin=0.1,
        noise_percentile=10.0,
    )


def test_mismatched_pairs_borrow_another_examples_response():
    exs = [Example(f"p{i}", f"r{i}", None, i) for i in range(5)]
    pairs, true_index = mismatched_pairs(exs, 3, seed=0)
    assert len(pairs) == len(true_index) == 3
    for pair, j in zip(pairs, true_index, strict=True):
        assert pair.response == exs[j].response  # exs[j] is the response's true pair
        assert pair.prompt != exs[j].prompt
    assert (pairs, true_index) == mismatched_pairs(exs, 3, seed=0)
    with pytest.raises(ValueError):
        mismatched_pairs(exs[:1], 3, seed=0)


def test_mismatched_pairs_n_one():
    exs = [Example(f"p{i}", f"r{i}", None, i) for i in range(5)]
    with pytest.raises(ValueError):
        mismatched_pairs(exs, 1, seed=0)


def test_classify():
    r = _classify(target=[0.5, 2.0, 3.0], reference=[0.45, 1.0, 3.5])
    assert r.status == "ok" and r.reason is None
    assert r.labels == ["known", "learnable", "noise"]
    assert r.noise_threshold == pytest.approx(3.2)
    assert (r.known_margin, r.noise_percentile) == (0.1, 10.0)
    assert r.fractions["known"] == pytest.approx(1 / 3)
    assert r.reliability == {
        "reference_below_target": pytest.approx(2 / 3),
        "min_reference_below_target": 0.5,
        "paired_discrimination": 1.0,
        "min_paired_discrimination": 0.6,
    }
    assert r.loss_quartiles["target"] == pytest.approx([1.25, 2.0, 2.5])
    assert r.loss_quartiles["reference"] == pytest.approx([0.725, 1.0, 2.25])
    summary = r.summary()
    json.dumps(summary)
    assert "labels" not in summary and summary["n_excluded"] == 0


def test_inconclusive_when_reference_is_not_below_target():
    # C1: a reference that scores answer style, not content, is above the target almost always.
    r = _classify(target=[0.5, 2.0, 3.0, 1.0], reference=[0.6, 2.5, 3.5, 0.9])
    assert r.status == "inconclusive"
    assert r.reason == (
        "the reference loss is below the target loss on only 25.0% of examples "
        "(gate: at least 50%)"
    )
    assert r.reliability["reference_below_target"] == 0.25
    assert sum(r.counts.values()) == 4  # the buckets are still recorded


def test_inconclusive_when_reference_cannot_tell_true_from_mismatched_pairs():
    true_pairs = [4.0] * 11 + [3.0] * 10  # mismatched > true on 10 of 21 pairs
    r = _classify([0.5, 2.0, 3.0], [0.45, 1.0, 3.5], true_pairs=true_pairs)
    assert r.status == "inconclusive"
    assert r.reliability["paired_discrimination"] == pytest.approx(
        (10 + 0.5 * 1) / 21  # 4.0 ties one mismatched loss: counted as half
    )
    assert r.reason == "paired discrimination 0.50 (gate: at least 0.60)"


def test_both_gates_give_both_reasons():
    r = _classify([1.0, 1.0], [2.0, 2.0], true_pairs=[9.0] * 21)
    assert r.status == "inconclusive"
    assert r.reason.startswith("the reference loss is below the target loss on only 0.0%")
    assert r.reason.endswith("; paired discrimination 0.00 (gate: at least 0.60)")


def test_nan_and_missing_losses_are_excluded_and_counted():
    # A NaN loss used to fall through every comparison and land in "learnable".
    r = _classify(
        target=[0.5, float("nan"), 2.0, 3.0, 1.0],
        reference=[0.45, 1.0, None, 3.5, 0.2],
        mismatched=MISMATCHED + [float("nan")],
        true_pairs=TRUE_PAIRS + [1.0],
    )
    assert r.labels == ["known", "excluded", "excluded", "noise", "learnable"]
    assert r.n_excluded == 2 and r.n_pairs_excluded == 1
    assert r.counts == {"known": 1, "learnable": 1, "noise": 1}
    assert r.fractions["known"] == pytest.approx(1 / 3)


def test_no_finite_losses_is_inconclusive():
    r = _classify([float("nan")], [1.0])
    assert r.status == "inconclusive" and r.reason == "no finite losses to compare"
    assert r.labels == ["excluded"] and r.n_excluded == 1
    r = _classify([1.0], [0.5], mismatched=[], true_pairs=[])
    assert r.status == "inconclusive" and r.reason == "no finite losses to compare"


def test_unavailable():
    r = novelty_unavailable("no reference model", 4)
    assert r.status == "unavailable" and r.reason == "no reference model"
    assert r.known_margin is None and r.noise_threshold is None and r.reliability is None
    json.dumps(r.summary())


def test_bucket_examples_two_per_bucket_clipped():
    exs = [
        Example("Short prompt", "Short answer", None, 1),
        Example("p2", "r2", None, 2),
        Example("p3", "r3", None, 3),
        Example("Long\nprompt  " + "word " * 40, "x" * 200, None, 7),
    ]
    out = bucket_examples(["known", "known", "known", "noise"], exs)
    assert [e["line"] for e in out["known"]] == [1, 2] and out["learnable"] == []
    (noisy,) = out["noise"]
    assert noisy["line"] == 7
    assert len(noisy["response"]) == 80 and noisy["response"].endswith("…")
    assert noisy["prompt"].startswith("Long prompt word") and len(noisy["prompt"]) == 80
    assert out["known"][0] == {"line": 1, "prompt": "Short prompt", "response": "Short answer"}
    assert bucket_examples(["excluded"], exs[:1]) == {"known": [], "learnable": [], "noise": []}


@pytest.mark.slow
@pytest.mark.parametrize("target", [m for m, info in REGISTRY.items() if info.reference_id])
def test_registry_reference_has_lower_loss_than_its_target(target):
    """A healthy reference (a larger sibling) has lower mean response loss than its target.
    Loads one model at a time; the 4B -> 8B pair needs about 17 GB of device memory."""
    from dvx.models import load_model, pick_device, release
    from dvx.scan.losses import response_losses

    reference = REGISTRY[target].reference_id
    sample = medmcqa_sample(16)
    mean = {}
    for model_id in (target, reference):
        model, tok = load_model(model_id, REGISTRY[model_id].revision, pick_device())
        mean[model_id] = float(np.mean(response_losses(model, tok, sample, max_len=512)))
        release(model)
    assert mean[reference] < mean[target], mean
