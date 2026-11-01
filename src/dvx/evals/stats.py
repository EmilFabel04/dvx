"""Bootstrap confidence intervals, paired deltas and the baseline check (pure numpy)."""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Estimate:
    value: float
    ci_low: float
    ci_high: float

    def includes_zero(self) -> bool:
        return self.ci_low <= 0.0 <= self.ci_high


@dataclass(frozen=True)
class BaselineCheck:
    ok: bool
    margin_pp: float
    per_benchmark_pp: dict[str, float]
    required_pp: float
    chance_pp: dict[str, float]


def _bootstrap(per_item: dict[str, np.ndarray], n_boot: int, seed: int) -> Estimate:
    """Macro-average (x100) of per-item values; resample within each benchmark independently."""
    rng = np.random.default_rng(seed)
    value = float(np.mean([v.mean() for v in per_item.values()]) * 100.0)
    boots = np.zeros(n_boot)
    for v in per_item.values():
        idx = rng.integers(0, len(v), size=(n_boot, len(v)))
        boots += v[idx].mean(axis=1)
    boots = boots / len(per_item) * 100.0
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # Guard against float rounding pushing the point estimate outside its interval.
    return Estimate(value, float(min(lo, value)), float(max(hi, value)))


def group_accuracy(correct: dict[str, np.ndarray], n_boot: int, seed: int) -> Estimate:
    """Macro-average accuracy in percent with a 95% bootstrap CI."""
    return _bootstrap({k: np.asarray(v, dtype=float) for k, v in correct.items()}, n_boot, seed)


def paired_delta(
    a: dict[str, np.ndarray], b: dict[str, np.ndarray], n_boot: int, seed: int
) -> Estimate:
    """Macro-average of (b - a) in percentage points; same resampled items for a and b."""
    if set(a) != set(b):
        raise ValueError(f"benchmark sets differ: {sorted(a)} vs {sorted(b)}")
    for k, va in a.items():
        if len(va) != len(b[k]):
            raise ValueError(f"benchmark {k!r}: lengths differ ({len(va)} vs {len(b[k])})")
    diffs = {k: np.asarray(b[k], dtype=float) - np.asarray(a[k], dtype=float) for k in a}
    return _bootstrap(diffs, n_boot, seed)


def chance_level(answers: Sequence[int], n_options: int) -> float:
    """What a no-skill model scores: max(1/k, the share of the most common answer)."""
    majority = Counter(answers).most_common(1)[0][1] / len(answers)
    return max(1.0 / n_options, majority)


def baseline_check(
    correct: dict[str, np.ndarray], chance: dict[str, float], min_margin: float
) -> BaselineCheck:
    """Mean over benchmarks of (accuracy - chance), in pp; ok if it reaches min_margin."""
    per = {k: float((np.mean(v) - chance[k]) * 100.0) for k, v in correct.items()}
    margin = float(np.mean(list(per.values())))
    required = float(min_margin * 100.0)
    chance_pp = {k: float(chance[k] * 100.0) for k in correct}
    return BaselineCheck(bool(margin >= required), margin, per, required, chance_pp)
