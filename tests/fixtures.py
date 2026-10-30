"""Shared test fixtures: a complete quick-run record dict."""

import copy


def _est(value: float, low: float, high: float) -> dict:
    return {"value": value, "ci_low": low, "ci_high": high}


def _groups(target: tuple, adjacent: tuple, side: tuple) -> dict:
    return {
        "target": _est(*target),
        "adjacent": _est(*adjacent),
        "side_effect": _est(*side),
    }


def _train() -> dict:
    return {
        "n_examples": 1000,
        "n_truncated": 0,
        "tokens_trained": 250_000,
        "steps": 120,
        "wall_seconds": 310.0,
        "est_flops": 1.5e15,
        "final_loss": 1.2,
        "seed": 0,
    }


def _deep_merge(base: dict, overrides: dict) -> dict:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def sample_record(**overrides) -> dict:
    """Build a complete, successful quick-run record.

    Keyword overrides are deep-merged into the default: dict values merge key by key
    (so `dataset={"n_total": 30000}` changes one field and
    `results={"exposure": {"status": "inconclusive"}}` changes one nested field);
    any non-dict value replaces. Merging cannot delete keys: to drop one, pop it from
    the returned dict.
    """
    base = {
        "schema_version": 1,
        "run_id": "20261030-210509-abcdef01",
        "created_at": "2026-10-30T21:05:09",
        "status": "complete",
        "engine_version": "0.1.0",
        "profile": "quick",
        "domain": "medical-qa",
        "suite_version": "1",
        "env": {"device": "cpu"},
        "dataset": {
            "path": "data/med.jsonl",
            "fingerprint": "abcdef0123456789" * 4,
            "n_total": 1000,
            "n_scanned": 1000,
            "n_trained": 1000,
            "declared_type": None,
            "declared_date_range": None,
        },
        "model": {
            "id": "Qwen/Qwen3-0.6B",
            "revision": "rev123",
            "params": 600_000_000,
            "family": "qwen3",
            "reference_id": None,
            "unseen_after": None,
        },
        "seeds": [0],
        "config": {"train": {"max_seq_len": 512}},
        "results": {
            "duplicates": {
                "n": 1000,
                "exact_dup": 20,
                "near_dup": 50,
                "exact_fraction": 0.02,
                "near_fraction": 0.05,
            },
            "contamination": {
                "n_flagged": 0,
                "flagged_lines": [],
                "by_benchmark": {},
                "ngram": 13,
            },
            "novelty": {
                "status": "ok",
                "reason": None,
                "counts": {"known": 500, "learnable": 400, "noise": 100},
                "fractions": {"known": 0.5, "learnable": 0.4, "noise": 0.1},
                "noise_threshold": 5.0,
                "known_margin": 0.1,
            },
            "exposure": {
                "status": "ok",
                "reason": None,
                "auc": 0.8,
                "threshold": 0.65,
                "seen_fraction": 0.3,
                "n_calibration": 200,
            },
            "baseline": {
                "ok": True,
                "margin_pp": 12.0,
                "per_benchmark_pp": {"medqa": 8.0, "medmcqa": 15.0, "pubmedqa": 13.0},
                "required_pp": 5.0,
            },
            "impact": {
                "accuracy": {
                    "base": _groups((40.0, 38.0, 42.0), (50.0, 48.0, 52.0), (60.0, 58.0, 62.0)),
                    "control": _groups((41.0, 39.0, 43.0), (50.0, 48.0, 52.0), (59.0, 57.0, 61.0)),
                    "tuned": _groups((44.2, 42.0, 46.4), (51.0, 49.0, 53.0), (57.0, 55.0, 59.0)),
                },
                "per_benchmark_accuracy": {
                    "base": {"medqa": 33.0, "medmcqa": 40.0, "pubmedqa": 55.0},
                    "control": {"medqa": 34.0, "medmcqa": 41.0, "pubmedqa": 56.0},
                    "tuned": {"medqa": 38.0, "medmcqa": 45.0, "pubmedqa": 58.0},
                },
                "delta_data_specific": _groups(
                    (3.2, 1.0, 5.4), (1.0, -1.0, 3.0), (-2.0, -4.0, 0.5)
                ),
                "delta_raw": _groups((4.2, 2.0, 6.4), (1.0, -1.0, 3.0), (-3.0, -5.0, -1.0)),
                "delta_control": _groups((1.0, -1.0, 3.0), (0.0, -2.0, 2.0), (-1.0, -3.0, 1.0)),
                "train": {"tuned": _train(), "control": _train()},
            },
        },
        "compute": {"steps": {}, "tokens_trained_total": 500_000, "est_flops_total": 3e15},
    }
    return _deep_merge(base, copy.deepcopy(overrides))
