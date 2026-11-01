"""Novelty buckets (known / learnable / noise), after RHO-LOSS, with a reliability gate."""

import random
from dataclasses import asdict, dataclass, replace
from typing import Literal

import numpy as np

from dvx.data import Example

BUCKETS = ("known", "learnable", "noise")
EXCLUDED = "excluded"  # label of an example whose target or reference loss is not finite
NO_LOSSES = "no finite losses to compare"
EXAMPLE_CHARS = 80


@dataclass(frozen=True)
class NoveltyResult:
    """Bucket labels per scanned example, plus how far the reference model can be trusted.

    `reliability` holds the two gate measures and their minimums: the share of examples
    whose reference loss is below the target loss, and paired discrimination, P(reference
    loss on a mismatched pair > reference loss on that response's true pair), ties
    counted as half. Below either minimum the status is "inconclusive"; the buckets are
    still computed and recorded.
    """

    status: Literal["ok", "inconclusive", "unavailable"]
    reason: str | None
    labels: list[str]
    counts: dict[str, int]
    fractions: dict[str, float]
    noise_threshold: float | None
    noise_percentile: float | None
    known_margin: float | None
    n_excluded: int
    n_pairs_excluded: int
    reliability: dict[str, float] | None
    loss_quartiles: dict[str, list[float]] | None

    def summary(self) -> dict:
        out = asdict(self)
        del out["labels"]
        return out


def mismatched_pairs(examples: list[Example], n: int, seed: int) -> tuple[list[Example], list[int]]:
    """Up to n examples, each with another sampled example's response.

    Returns the pairs and, for each, the index into `examples` of the example whose
    response it borrowed: that response's true pair.
    """
    m = min(n, len(examples))
    if m < 2:
        raise ValueError(f"need at least 2 examples to build mismatched pairs, got {m}")
    picked = random.Random(seed).sample(range(len(examples)), m)
    true_index = [picked[(i + 1) % m] for i in range(m)]
    pairs = [
        replace(examples[i], response=examples[j].response)
        for i, j in zip(picked, true_index, strict=True)
    ]
    return pairs, true_index


def _losses(values) -> np.ndarray:
    """Float array; None (a non-finite loss stored as JSON null) becomes NaN."""
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def _quartiles(values: np.ndarray) -> list[float]:
    return [float(q) for q in np.percentile(values, [25, 50, 75])]


def classify(
    target: list[float],
    reference: list[float],
    mismatched_reference: list[float],
    mismatched_true_reference: list[float],
    known_margin: float,
    noise_percentile: float,
    min_reference_below_target: float = 0.5,
    min_paired_discrimination: float = 0.6,
) -> NoveltyResult:
    """Bucket each example by its target and reference losses, then gate on reliability.

    An example is noise if its reference loss reaches the `noise_percentile` of the
    mismatched-pair reference losses, known if its target loss is within `known_margin`
    of the reference loss, else learnable. Examples (and mismatched pairs) with a NaN or
    missing loss are excluded and counted. `mismatched_true_reference[i]` is the reference
    loss of the true pair of mismatched pair i.
    """
    t, r = _losses(target), _losses(reference)
    m, mt = _losses(mismatched_reference), _losses(mismatched_true_reference)
    if t.shape != r.shape or m.shape != mt.shape:
        raise ValueError("loss lists must pair up: target with reference, mismatched with true")
    ok = np.isfinite(t) & np.isfinite(r)
    pair_ok = np.isfinite(m) & np.isfinite(mt)
    excluded = int((~ok).sum()), int((~pair_ok).sum())
    params = {"known_margin": float(known_margin), "noise_percentile": float(noise_percentile)}
    if not ok.any() or not pair_ok.any():
        return NoveltyResult(
            status="inconclusive",
            reason=NO_LOSSES,
            labels=[EXCLUDED] * len(t),
            counts={},
            fractions={},
            noise_threshold=None,
            **params,
            n_excluded=len(t),
            n_pairs_excluded=excluded[1],
            reliability=None,
            loss_quartiles=None,
        )

    threshold = float(np.percentile(m[pair_ok], noise_percentile))
    labels = []
    for valid, tl, rl in zip(ok, t, r, strict=True):
        if not valid:
            labels.append(EXCLUDED)
        elif rl >= threshold:
            labels.append("noise")
        elif tl - rl <= known_margin:
            labels.append("known")
        else:
            labels.append("learnable")
    counts = {b: labels.count(b) for b in BUCKETS}
    n = int(ok.sum())
    fractions = {b: counts[b] / n for b in BUCKETS}

    mv, mtv = m[pair_ok], mt[pair_ok]
    reliability = {
        "reference_below_target": float((r[ok] < t[ok]).mean()),
        "min_reference_below_target": float(min_reference_below_target),
        "paired_discrimination": float((mv > mtv).mean() + 0.5 * (mv == mtv).mean()),
        "min_paired_discrimination": float(min_paired_discrimination),
    }
    reasons = []
    if reliability["reference_below_target"] < min_reference_below_target:
        reasons.append(
            "the reference loss is below the target loss on only "
            f"{reliability['reference_below_target']:.1%} of examples "
            f"(gate: at least {min_reference_below_target:.0%})"
        )
    if reliability["paired_discrimination"] < min_paired_discrimination:
        reasons.append(
            f"paired discrimination {reliability['paired_discrimination']:.2f} "
            f"(gate: at least {min_paired_discrimination:.2f})"
        )
    return NoveltyResult(
        "inconclusive" if reasons else "ok",
        "; ".join(reasons) or None,
        labels,
        counts,
        fractions,
        threshold,
        **params,
        n_excluded=excluded[0],
        n_pairs_excluded=excluded[1],
        reliability=reliability,
        loss_quartiles={"target": _quartiles(t[ok]), "reference": _quartiles(r[ok])},
    )


def _clip(text: str, width: int) -> str:
    """One line, at most `width` characters (an ellipsis marks a cut)."""
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


def bucket_examples(
    labels: list[str], examples: list[Example], per_bucket: int = 2, width: int = EXAMPLE_CHARS
) -> dict[str, list[dict]]:
    """The first `per_bucket` examples of each bucket, in scan order, clipped for display."""
    out: dict[str, list[dict]] = {b: [] for b in BUCKETS}
    for label, ex in zip(labels, examples, strict=True):
        if label in out and len(out[label]) < per_bucket:
            prompt, response = _clip(ex.prompt, width), _clip(ex.response, width)
            out[label].append({"line": ex.line, "prompt": prompt, "response": response})
    return out


def novelty_unavailable(reason: str, n: int) -> NoveltyResult:
    return NoveltyResult(
        "unavailable", reason, ["unknown"] * n, {}, {}, None, None, None, 0, 0, None, None
    )
