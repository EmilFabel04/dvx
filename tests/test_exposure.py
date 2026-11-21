import numpy as np
import pytest

from dvx.scan.exposure import (
    ExposureResult,
    assess_exposure,
    auc,
    exposure_inconclusive,
    min_k_score,
)


def test_min_k():
    assert min_k_score(np.array([-1.0, -2.0, -3.0, -4.0, -5.0]), k=0.4) == -4.5
    assert np.isnan(min_k_score(np.array([])))


def test_min_k_returns_plain_float():
    assert type(min_k_score(np.array([-1.0, -2.0]))) is float


def test_auc():
    assert auc([3, 4], [1, 2]) == 1.0
    assert auc([1], [1]) == 0.5
    assert auc([1, 2], [3, 4]) == 0.0


def test_auc_empty_is_nan():
    assert np.isnan(auc([], [1.0]))


def test_inconclusive_low_auc():
    r = assess_exposure([0.0], [0, 1, 0, 1], [0, 1, 1, 0])
    assert r.status == "inconclusive" and "AUC" in r.reason
    assert r.reason == "calibration AUC 0.50 below 0.6"
    assert r.auc == 0.5 and r.seen_fraction is None and r.n_calibration == 8


def test_ok():
    r = assess_exposure(
        [-1.0, -3.0, -1.05, float("nan")],
        [-1.0, -1.1, -0.9],
        [-3.0, -3.1, -2.9],
    )
    assert r.status == "ok" and r.auc == 1.0
    assert r.seen_fraction == pytest.approx(2 / 3)
    assert r.threshold == pytest.approx(-1.1)
    assert r.reason is None and r.n_calibration == 6
    assert type(r.auc) is float and type(r.threshold) is float


def test_no_calibration_scores_is_inconclusive():
    r = assess_exposure([-1.0], [], [-2.0])
    assert r.status == "inconclusive" and r.auc is None


def test_no_dataset_scores_is_inconclusive():
    r = assess_exposure([float("nan")], [-1.0], [-3.0])
    assert r.status == "inconclusive" and r.seen_fraction is None


def test_exposure_inconclusive():
    r = exposure_inconclusive("calibration unavailable")
    assert r == ExposureResult("inconclusive", "calibration unavailable", None, None, None, 0)
