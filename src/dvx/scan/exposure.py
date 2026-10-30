"""Membership-inference exposure estimate, calibrated on known seen/unseen texts.

Pure functions over numbers: the pipeline computes per-text scores (for example
``min_k_score`` of ``dvx.scan.losses.token_logprobs``) and passes them in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class ExposureResult:
    status: Literal["ok", "inconclusive"]
    reason: str | None
    auc: float | None
    threshold: float | None
    seen_fraction: float | None
    n_calibration: int


def min_k_score(logprobs: np.ndarray, k: float = 0.2) -> float:
    """Mean of the lowest ceil(k * n) token log-probabilities; nan when empty."""
    arr = np.asarray(logprobs, dtype=float)
    n = arr.size
    if n == 0:
        return float("nan")
    m = max(1, math.ceil(k * n))
    return float(np.sort(arr)[:m].mean())


def auc(pos, neg) -> float:
    """Mann-Whitney AUC: P(pos > neg) with ties counted as 0.5; nan if either is empty."""
    p = np.asarray(pos, dtype=float)
    q = np.asarray(neg, dtype=float)
    if p.size == 0 or q.size == 0:
        return float("nan")
    diff = p[:, None] - q[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)


def exposure_inconclusive(reason: str) -> ExposureResult:
    return ExposureResult("inconclusive", reason, None, None, None, 0)


def _finite(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[~np.isnan(arr)]


def assess_exposure(dataset_scores, seen_scores, unseen_scores, min_auc: float = 0.6) -> ExposureResult:
    seen = _finite(seen_scores)
    unseen = _finite(unseen_scores)
    data = _finite(dataset_scores)
    n_cal = int(seen.size + unseen.size)
    if seen.size == 0 or unseen.size == 0:
        return ExposureResult("inconclusive", "no calibration scores", None, None, None, n_cal)
    cal_auc = auc(seen, unseen)
    if cal_auc < min_auc:
        return ExposureResult(
            "inconclusive", f"calibration AUC {cal_auc:.2f} below {min_auc}",
            cal_auc, None, None, n_cal,
        )
    candidates = np.unique(np.concatenate([seen, unseen]))
    youden = [(seen >= t).mean() - (unseen >= t).mean() for t in candidates]
    threshold = float(candidates[int(np.argmax(youden))])
    if data.size == 0:
        return ExposureResult(
            "inconclusive", "no finite dataset scores", cal_auc, threshold, None, n_cal
        )
    return ExposureResult(
        "ok", None, cal_auc, threshold, float((data >= threshold).mean()), n_cal
    )
