from fixtures import sample_record

from dvx.report import fmt_delta, render_report, render_summary


def test_fmt_delta_positive_and_zero():
    assert fmt_delta({"value": 3.2, "ci_low": 1.0, "ci_high": 5.4}) == (
        "+3.2 pp (95% CI +1.0 to +5.4)"
    )
    assert fmt_delta({"value": 0.5, "ci_low": -1.0, "ci_high": 2.0}) == (
        "+0.5 pp (95% CI -1.0 to +2.0) — not distinguishable from zero"
    )


def test_fmt_delta_marks_a_regression():
    assert fmt_delta({"value": -3.8, "ci_low": -7.5, "ci_high": -0.4}) == (
        "-3.8 pp (95% CI -7.5 to -0.4) — regression"
    )


def test_header_and_context():
    report = render_report(sample_record())
    assert report.startswith("# Data Impact Report: Qwen/Qwen3-0.6B-Base on data/med.jsonl")
    assert "medical-qa" in report and "abcdef012345" in report and "rev123" in report
    assert "abcdef0123456" not in report


def test_sample_notes():
    rec = sample_record(dataset={"n_total": 30000, "n_scanned": 20000, "n_trained": 5000})
    r = render_report(rec)
    assert "Measured on a sample of 20,000 of 30,000 examples." in r
    assert "Impact measured by training on 5,000 of 30,000 examples." in r
    assert "Measured on a sample" not in render_report(sample_record())
    assert "Impact measured by training on" not in render_report(sample_record())


def test_no_training_note_without_impact():
    for status in ("scan_only", "stopped_baseline"):
        rec = sample_record(
            status=status, dataset={"n_total": 30000, "n_scanned": 20000, "n_trained": 5000}
        )
        rec["results"].pop("impact")
        for text in (render_report(rec), render_summary(rec)):
            assert "Measured on a sample of 20,000 of 30,000 examples." in text
            assert "Impact measured by training on" not in text


def test_headline_then_side_effects_table():
    r = render_report(sample_record())
    headline = "Data-specific gain on target benchmarks: +3.2 pp (95% CI +1.0 to +5.4)"
    assert headline in r
    assert "## Side effects" in r
    assert r.index(headline) < r.index("## Side effects")
    side = r[r.index("## Side effects") :]
    assert "-2.0 pp (95% CI -4.0 to +0.5) — not distinguishable from zero" in side
    assert "-3.0 pp (95% CI -5.0 to -1.0) — regression" in side  # raw, marked in the table
    for group in ("target", "adjacent", "side_effect"):
        assert group in side


def test_side_effects_table_shows_the_format_effect():
    r = render_report(sample_record())
    assert "| Group | Data-specific (tuned − control) | Raw (tuned − base) | " in r
    assert "Format effect (control − base) |" in r
    assert (
        "| side_effect | -2.0 pp (95% CI -4.0 to +0.5) — not distinguishable from zero "
        "| -3.0 pp (95% CI -5.0 to -1.0) — regression "
        "| -1.0 pp (95% CI -3.0 to +1.0) — not distinguishable from zero |"
    ) in r


def _est(value, low, high):
    return {"value": value, "ci_low": low, "ci_high": high}


REGRESSION = (
    "Regression vs. control: adjacent -3.8 pp (95% CI -7.5 to -0.4; one training seed). "
    "Versus the untuned base: +18.0 pp."
)


def _regressed(**kw):
    return sample_record(
        results={
            "impact": {
                "delta_data_specific": {"adjacent": _est(-3.8, -7.5, -0.4)},
                "delta_raw": {"adjacent": _est(18.0, 13.9, 22.4)},
            }
        },
        **kw,
    )


def test_regression_called_out_next_to_the_headline():
    r = render_report(_regressed())
    headline = r.index("Data-specific gain on target benchmarks")
    assert headline < r.index(REGRESSION) < r.index("## Side effects")
    assert REGRESSION in render_summary(_regressed())
    for text in (render_report(sample_record()), render_summary(sample_record())):
        assert "Regression vs. control" not in text  # side_effect's CI reaches +0.5


def test_regression_line_counts_training_seeds():
    r = render_report(_regressed(seeds=[0, 1, 2]))
    assert "(95% CI -7.5 to -0.4; 3 training seeds)" in r


def test_summary_has_one_line_per_side_effect_group():
    s = render_summary(_regressed()).splitlines()
    assert "Side effects (data-specific, tuned − control):" in s
    assert "  adjacent: -3.8 pp (95% CI -7.5 to -0.4) — regression" in s
    assert (
        "  side_effect: -2.0 pp (95% CI -4.0 to +0.5) — not distinguishable from zero"
    ) in s


def test_per_benchmark_table():
    r = render_report(sample_record())
    assert "| medqa | 33.0 | 34.0 | 38.0 |" in r


def test_novelty_ok_and_unavailable():
    r = render_report(sample_record())
    assert "Known to the model: 50.0% · learnable: 40.0% · noise: 10.0%" in r
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


NOVELTY_DETAIL = (
    (
        "Reliability: the reference loss is below the target loss on 72.0% of examples "
        "(gate: at least 50%); paired discrimination 0.81 (gate: at least 0.60)."
    ),
    (
        "Reference model: Qwen/Qwen3-1.7B-Base (revision refrev456). Thresholds: known margin "
        "0.10 nats per token; noise threshold 5.00 nats per token (the 10th percentile of "
        "mismatched-pair reference loss)."
    ),
    (
        "Loss quartiles (nats per response token): target 1.20 / 2.30 / 3.10; "
        "reference 1.00 / 2.00 / 2.90."
    ),
    '- known, line 3: "Which nerve?" → "B. Ulnar"',
    '- known, line 9: "Drug of choice?" → "A. Penicillin"',
    '- learnable, line 4: "Rare sign?" → "C. Kehr"',
)


def test_novelty_shows_reliability_reference_thresholds_quartiles_examples():
    r = render_report(sample_record())
    for line in NOVELTY_DETAIL:
        assert line in r
    assert r.index("Known to the model") < r.index("Reliability:") < r.index("- known, line 3")


def test_inconclusive_novelty_hides_the_mix_but_shows_its_reliability():
    reason = (
        "the reference loss is below the target loss on only 0.7% of examples "
        "(gate: at least 50%)"
    )
    rec = sample_record(
        results={
            "novelty": {
                "status": "inconclusive",
                "reason": reason,
                "reliability": {"reference_below_target": 0.007},
            }
        }
    )
    r = render_report(rec)
    assert f"Novelty: inconclusive ({reason})." in r
    assert "Known to the model" not in r
    assert "below the target loss on 0.7% of examples (gate: at least 50%)" in r
    assert "Example items, as the unreliable rule buckets them:" in r
    for line in NOVELTY_DETAIL[1:]:
        assert line in r


def test_novelty_counts_excluded_examples():
    rec = sample_record(results={"novelty": {"n_excluded": 3}})
    assert "3 examples were excluded from novelty (non-finite loss)." in render_report(rec)
    assert "excluded from novelty" not in render_report(sample_record())


def test_schema_1_novelty_still_renders():
    rec = sample_record()
    for key in ("reliability", "loss_quartiles", "reference", "examples", "n_excluded"):
        rec["results"]["novelty"].pop(key)
    r = render_report(rec)
    assert "Known to the model: 50.0%" in r and "Reliability:" not in r


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


def test_truncation_shows_both_arms():
    rec = sample_record(results={"impact": {"train": {"tuned": {"n_truncated": 7}}}})
    assert "Training examples truncated to 512 tokens: tuned 7, control 0." in render_report(rec)
    rec = sample_record(results={"impact": {"train": {"control": {"n_truncated": 1}}}})
    assert "Training examples truncated to 512 tokens: tuned 0, control 1." in render_report(rec)
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


def test_counts_agree_in_number():
    one = sample_record(results={"contamination": {"n_flagged": 1, "by_benchmark": {"medqa": 1}}})
    assert "WARNING: 1 example overlaps with evaluation items (medqa: 1)." in render_report(one)
    rec = sample_record(dataset={"n_total": 2, "n_scanned": 1, "n_trained": 1})
    r = render_report(rec)
    assert "Measured on a sample of 1 of 2 examples." in r
    assert "Impact measured by training on 1 of 2 examples." in r


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
    assert "Caveats" not in r


def test_baseline_at_or_below_chance_note():
    rec = sample_record(
        results={
            "baseline": {"per_benchmark_pp": {"medqa": -1.5, "medmcqa": 15.0, "pubmedqa": 0.0}}
        }
    )
    r = render_report(rec)
    assert (
        "Note: medqa is at or below chance for the base model (-1.5 pp vs. chance 30.0%); "
        "gains there are unreliable."
    ) in r
    assert "Note: pubmedqa is at or below chance for the base model (+0.0 pp vs. chance 55.2%)" in r
    assert "Note: medmcqa" not in r
    assert "at or below chance" not in render_report(sample_record())


def test_chance_notes_rendered_once():
    rec = sample_record(results={"baseline": {"per_benchmark_pp": {"medqa": -1.5}}})
    assert render_report(rec).count("Note: medqa is at or below chance") == 1
    assert render_summary(rec).count("Note: medqa is at or below chance") == 1
    rec = sample_record(
        status="stopped_baseline", results={"baseline": {"per_benchmark_pp": {"medqa": -1.5}}}
    )
    rec["results"].pop("impact")
    r = render_report(rec)
    assert r.count("Note: medqa is at or below chance") == 1
    assert r.index("## Baseline") < r.index("Note: medqa")


def test_baseline_section_states_chance_per_benchmark():
    r = render_report(sample_record())
    assert (
        "Chance is the larger of 1/k and the share of the most common answer among the eval "
        "items: medqa 30.0%, medmcqa 37.5%, pubmedqa 55.2%."
    ) in r


def test_baseline_note_ignores_non_target_benchmarks():
    rec = sample_record(results={"baseline": {"per_benchmark_pp": {"arc_challenge": -3.0}}})
    assert "Note: arc_challenge" not in render_report(rec)


def test_caveats_only_with_impact():
    r = render_report(sample_record())
    assert "## Caveats" in r
    assert (
        "Quick profile: one training seed; confidence intervals cover question sampling "
        "only, not training randomness."
    ) in r
    assert (
        "The control is generic instruction data; part of the raw gain can come from "
        "learning the answer format."
    ) in r
    for status in ("scan_only", "stopped_baseline"):
        rec = sample_record(status=status)
        rec["results"].pop("impact")
        r = render_report(rec)
        assert "Caveats" not in r
        assert "Quick profile:" not in r and "generic instruction data" not in r


def test_summary_has_headline_and_warnings():
    rec = sample_record(results={"contamination": {"n_flagged": 3, "by_benchmark": {"medqa": 3}}})
    s = render_summary(rec)
    assert s.startswith("Data Impact Report: Qwen/Qwen3-0.6B-Base on data/med.jsonl")
    assert "Data-specific gain on target benchmarks: +3.2 pp" in s
    assert "Side effects (data-specific, tuned − control):" in s
    assert "WARNING: 3 examples overlap" in s
    assert "| " not in s


def test_eval_truncation_reported_before_gains():
    counts = {"medqa": 0, "medmcqa": 0, "pubmedqa": 12, "arc_challenge": 1}
    rec = sample_record(evaluation={"n_truncated": counts})
    line = (
        "Evaluation prompts left-truncated to 3072 tokens (their earliest shots cut): "
        "pubmedqa 12, arc_challenge 1."
    )
    r = render_report(rec)
    assert line in r and r.index(line) < r.index("## Side effects")
    assert line in render_summary(rec)
    assert "left-truncated" not in render_report(sample_record())
    rec = sample_record(status="stopped_baseline", evaluation={"n_truncated": counts})
    rec["results"].pop("impact")
    assert line in render_report(rec)


def test_context_line_names_the_eval_protocol():
    r = render_report(sample_record())
    assert "suite v2, plain-text 5-shot log-likelihood letter scoring" in r


def test_contamination_warning_precedes_gains():
    rec = sample_record(results={"contamination": {"n_flagged": 3, "by_benchmark": {"medqa": 3}}})
    r = render_report(rec)
    assert r.index("Data-specific gain") < r.index("WARNING") < r.index("## Side effects")


def test_truncation_precedes_gains():
    rec = sample_record(results={"impact": {"train": {"tuned": {"n_truncated": 7}}}})
    r = render_report(rec)
    assert r.index("Training examples truncated") < r.index("## Side effects")


def test_chance_note_next_to_benchmark_table():
    rec = sample_record(results={"baseline": {"per_benchmark_pp": {"medqa": -1.5}}})
    r = render_report(rec)
    note = r.index("Note: medqa is at or below chance")
    assert r.index("| medqa |") < note < r.index("## Data checks")


def test_summary_has_chance_notes_and_sample_notes():
    rec = sample_record(
        dataset={"n_total": 30000, "n_scanned": 20000, "n_trained": 5000},
        results={"baseline": {"per_benchmark_pp": {"medqa": -1.5}}},
    )
    s = render_summary(rec)
    assert "Note: medqa is at or below chance for the base model (-1.5 pp vs. chance" in s
    assert "Measured on a sample of 20,000 of 30,000 examples." in s
    assert "Impact measured by training on 5,000 of 30,000 examples." in s
