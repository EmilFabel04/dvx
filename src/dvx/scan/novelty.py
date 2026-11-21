"""Novelty buckets (known / learnable / noise), after RHO-LOSS."""

import random
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from dvx.data import Example

BUCKETS = ("known", "learnable", "noise")


@dataclass(frozen=True)
class NoveltyResult:
    status: Literal["ok", "unavailable"]
    reason: str | None
    labels: list[str]
    counts: dict[str, int]
    fractions: dict[str, float]
    noise_threshold: float | None
    known_margin: float

    def summary(self) -> dict:
        return {
            "status": self.status,
            "reason": self.reason,
            "counts": dict(self.counts),
            "fractions": dict(self.fractions),
            "noise_threshold": self.noise_threshold,
            "known_margin": self.known_margin,
        }


def mismatched_pairs(examples: list[Example], n: int, seed: int) -> list[Example]:
    m = min(n, len(examples))
    if m < 2:
        raise ValueError(f"need at least 2 examples to build mismatched pairs, got {m}")
    sample = random.Random(seed).sample(examples, m)
    return [replace(ex, response=sample[(i + 1) % m].response) for i, ex in enumerate(sample)]


def classify(
    target: list[float],
    reference: list[float],
    mismatched_reference: list[float],
    known_margin: float,
    noise_percentile: float,
) -> NoveltyResult:
    threshold = float(np.percentile(mismatched_reference, noise_percentile))
    labels = []
    for t, r in zip(target, reference, strict=True):
        if r >= threshold:
            labels.append("noise")
        elif t - r <= known_margin:
            labels.append("known")
        else:
            labels.append("learnable")
    counts = {b: labels.count(b) for b in BUCKETS}
    n = len(labels)
    fractions = {b: (counts[b] / n if n else 0.0) for b in BUCKETS}
    return NoveltyResult("ok", None, labels, counts, fractions, threshold, float(known_margin))


def novelty_unavailable(reason: str, n: int) -> NoveltyResult:
    return NoveltyResult("unavailable", reason, ["unknown"] * n, {}, {}, None, 0.0)
