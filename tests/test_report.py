from fixtures import sample_record

from dvx.report import fmt_delta, render_report, render_summary


def test_fmt_delta_positive_and_zero():
    assert fmt_delta({"value": 3.2, "ci_low": 1.0, "ci_high": 5.4}) == (
        "+3.2 pp (95% CI +1.0 to +5.4)"
    )
    assert fmt_delta({"value": 0.5, "ci_low": -1.0, "ci_high": 2.0}) == (
        "+0.5 pp (95% CI -1.0 to +2.0) — not distinguishable from zero"
    )


def test_header_and_context():
    report = render_report(sample_record())
    assert report.startswith("# Data Impact Report: Qwen/Qwen3-0.6B on data/med.jsonl")
    assert "medical-qa" in report and "abcdef012345" in report and "rev123" in report
    assert "abcdef0123456" not in report


def test_sample_notes():
    rec = sample_record(dataset={"n_total": 30000, "n_scanned": 20000, "n_trained": 5000})
    r = render_report(rec)
    assert "Measured on a sample of 20,000 of 30,000 examples." in r
    assert "Impact measured by training on 5,000 of 30,000 examples." in r
    assert "Measured on a sample" not in render_report(sample_record())
    assert "Impact measured by training on" not in render_report(sample_record())


def test_headline_then_side_effects_table():
    r = render_report(sample_record())
    headline = "Data-specific gain on target benchmarks: +3.2 pp (95% CI +1.0 to +5.4)"
    assert headline in r
    assert "## Side effects" in r
    assert r.index(headline) < r.index("## Side effects")
    side = r[r.index("## Side effects") :]
    assert "-2.0 pp (95% CI -4.0 to +0.5) — not distinguishable from zero" in side
    assert "-3.0 pp (95% CI -5.0 to -1.0)" in side
    for group in ("target", "adjacent", "side_effect"):
        assert group in side


def test_per_benchmark_table():
    r = render_report(sample_record())
    assert "| medqa | 33.0 | 34.0 | 38.0 |" in r


def test_novelty_ok_and_unavailable():
    r = render_report(sample_record())
    assert "50.0%" in r and "40.0%" in r and "10.0%" in r
    rec = sample_record(
        results={
            "novelty": {
                "status": "unavailable",
                "reason": "no reference model",
                "counts": {},
                "fractions": {},
            }
        }
    )
    assert "Novelty: unavailable (no reference model)" in render_report(rec)


def test_exposure_ok_and_inconclusive():
    r = render_report(sample_record())
    assert "About 30% of examples look previously seen by this model (calibration AUC 0.80)." in r
    assert (
        "Calibration compares old vs. new Wikipedia articles; topic drift can inflate the AUC."
    ) in r
    rec = sample_record(
        results={"exposure": {"status": "inconclusive", "reason": "AUC below threshold"}}
    )
    r = render_report(rec)
    assert "Prior exposure: inconclusive (AUC below threshold)." in r
    assert "look previously seen" not in r


def test_duplicates():
    r = render_report(sample_record())
    assert "Exact duplicates: 2.0%" in r and "near duplicates: 5.0%" in r


def test_truncation():
    rec = sample_record(results={"impact": {"train": {"tuned": {"n_truncated": 7}}}})
    assert "7 training examples were truncated to 512 tokens." in render_report(rec)
    assert "truncated" not in render_report(sample_record())


def test_contamination_warning():
    rec = sample_record(
        results={
            "contamination": {
                "n_flagged": 3,
                "flagged_lines": [1, 2, 3],
                "by_benchmark": {"medqa": 3},
            }
        }
    )
    assert (
        "WARNING: 3 examples overlap with evaluation items (medqa: 3). "
        "Gains on those benchmarks may be inflated."
    ) in render_report(rec)
    assert "WARNING" not in render_report(sample_record())


def test_baseline_stop():
    rec = sample_record(
        status="stopped_baseline",
        results={"baseline": {"ok": False, "margin_pp": 1.0, "required_pp": 5.0}},
    )
    rec["results"].pop("impact")
    r = render_report(rec)
    assert (
        "Stopped: the base model is not clearly above chance on the target benchmarks "
        "(margin 1.0 pp, need 5.0 pp). Try a larger model."
    ) in r
    assert "Side effects" not in r and "Data-specific gain" not in r
    assert "Caveats" in r


def test_baseline_at_or_below_chance_note():
    rec = sample_record(
        results={
            "baseline": {"per_benchmark_pp": {"medqa": -1.5, "medmcqa": 15.0, "pubmedqa": 0.0}}
        }
    )
    r = render_report(rec)
    assert (
        "Note: medqa is at or below chance for the base model (-1.5 pp vs. chance); "
        "gains there are unreliable."
    ) in r
    assert "Note: pubmedqa is at or below chance for the base model (+0.0 pp vs. chance)" in r
    assert "Note: medmcqa" not in r
    assert "at or below chance" not in render_report(sample_record())


def test_baseline_note_ignores_non_target_benchmarks():
    rec = sample_record(results={"baseline": {"per_benchmark_pp": {"arc_challenge": -3.0}}})
    assert "Note: arc_challenge" not in render_report(rec)


def test_caveats_always_present():
    for rec in (sample_record(), sample_record(status="stopped_baseline")):
        r = render_report(rec)
        assert "## Caveats" in r
        assert (
            "Quick profile: one training seed; confidence intervals cover question sampling "
            "only, not training randomness."
        ) in r
        assert (
            "The control is generic instruction data; part of the raw gain can come from "
            "learning the answer format."
        ) in r


def test_summary_has_headline_and_warnings():
    rec = sample_record(results={"contamination": {"n_flagged": 3, "by_benchmark": {"medqa": 3}}})
    s = render_summary(rec)
    assert s.startswith("Data Impact Report: Qwen/Qwen3-0.6B on data/med.jsonl")
    assert "Data-specific gain on target benchmarks: +3.2 pp" in s
    assert "Side effects" in s
    assert "WARNING: 3 examples overlap" in s
    assert "| " not in s
