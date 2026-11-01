"""Render a run record (dict) as a Markdown report and a terminal summary.

Pure functions: dict in, str out. The pipeline and CLI write the files.
"""

from dvx.evals.registry import get_suite

GROUPS = ("target", "adjacent", "side_effect")

CAVEATS = (
    (
        "Quick profile: one training seed; confidence intervals cover question sampling only, "
        "not training randomness."
    ),
    (
        "The control is generic instruction data; part of the raw gain can come from learning "
        "the answer format."
    ),
)


def fmt_delta(e: dict) -> str:
    """Format an Estimate of a delta, in percentage points with its 95% CI."""
    text = f"{e['value']:+.1f} pp (95% CI {e['ci_low']:+.1f} to {e['ci_high']:+.1f})"
    if e["ci_low"] <= 0 <= e["ci_high"]:
        text += " — not distinguishable from zero"
    return text


def _header(record: dict) -> str:
    return f"Data Impact Report: {record['model']['id']} on {record['dataset']['path']}"


def _headline(record: dict) -> str | None:
    impact = record["results"].get("impact")
    if not impact:
        return None
    return "Data-specific gain on target benchmarks: " + fmt_delta(
        impact["delta_data_specific"]["target"]
    )


def _stop_line(record: dict) -> str | None:
    if record["status"] != "stopped_baseline":
        return None
    b = record["results"]["baseline"]
    return (
        "Stopped: the base model is not clearly above chance on the target benchmarks "
        f"(margin {b['margin_pp']:.1f} pp, need {b['required_pp']:.1f} pp). Try a larger model."
    )


def _contamination_warning(record: dict) -> str | None:
    c = record["results"].get("contamination")
    if not c or c["n_flagged"] <= 0:
        return None
    by = ", ".join(f"{name}: {n}" for name, n in c["by_benchmark"].items())
    return (
        f"WARNING: {c['n_flagged']:,} examples overlap with evaluation items ({by}). "
        "Gains on those benchmarks may be inflated."
    )


def _truncation_line(record: dict) -> str | None:
    impact = record["results"].get("impact")
    if not impact:
        return None
    n = impact["train"]["tuned"]["n_truncated"]
    if n <= 0:
        return None
    return f"{n:,} training examples were truncated to {record['config']['train']['max_seq_len']} tokens."


def _eval_truncation_line(record: dict) -> str | None:
    ev = record.get("evaluation")
    cut = {b: n for b, n in (ev or {}).get("n_truncated", {}).items() if n > 0}
    if not cut:
        return None
    by = ", ".join(f"{b} {n:,}" for b, n in cut.items())
    return (
        f"Evaluation prompts left-truncated to {ev['max_len']} tokens "
        f"(their earliest shots cut): {by}."
    )


def _sample_notes(record: dict) -> list[str]:
    d = record["dataset"]
    notes = []
    if d["n_scanned"] < d["n_total"]:
        notes.append(f"Measured on a sample of {d['n_scanned']:,} of {d['n_total']:,} examples.")
    if "impact" in record["results"] and d["n_trained"] < d["n_total"]:
        notes.append(
            f"Impact measured by training on {d['n_trained']:,} of {d['n_total']:,} examples."
        )
    return notes


def _context_line(record: dict) -> str:
    m, d = record["model"], record["dataset"]
    ev = record.get("evaluation")
    suite = f"suite v{record['suite_version']}" + (f", {ev['protocol']}" if ev else "")
    return (
        f"Domain: {record['domain']} ({suite}) · "
        f"profile: {record['profile']} · dataset fingerprint: {d['fingerprint'][:12]} · "
        f"model revision: {m['revision']} · parameters: {m['params']:,} · "
        f"seeds: {', '.join(str(s) for s in record['seeds'])}"
    )


def _side_effects_table(impact: dict) -> list[str]:
    lines = [
        "| Group | Data-specific (tuned − control) | Raw (tuned − base) |",
        "|---|---|---|",
    ]
    for g in GROUPS:
        lines.append(
            f"| {g} | {fmt_delta(impact['delta_data_specific'][g])} "
            f"| {fmt_delta(impact['delta_raw'][g])} |"
        )
    return lines


def _per_benchmark_table(impact: dict) -> list[str]:
    acc = impact["per_benchmark_accuracy"]
    lines = ["| Benchmark | Base | Control | Tuned |", "|---|---|---|---|"]
    for bench in acc["base"]:
        lines.append(
            f"| {bench} | {acc['base'][bench]:.1f} | {acc['control'][bench]:.1f} "
            f"| {acc['tuned'][bench]:.1f} |"
        )
    return lines


def _novelty_lines(novelty: dict) -> list[str]:
    if novelty["status"] != "ok":
        return [f"Novelty: unavailable ({novelty['reason']})"]
    f = novelty["fractions"]
    return [
        (
            f"Known to the model: {f['known']:.1%} · learnable: {f['learnable']:.1%} "
            f"· noise: {f['noise']:.1%}"
        )
    ]


def _exposure_lines(exposure: dict) -> list[str]:
    if exposure["status"] != "ok":
        return [f"Prior exposure: inconclusive ({exposure['reason']})."]
    return [
        (
            f"About {exposure['seen_fraction']:.0%} of examples look previously seen by this model "
            f"(calibration AUC {exposure['auc']:.2f})."
        ),
        "Calibration compares old vs. new Wikipedia articles; topic drift can inflate the AUC.",
    ]


def _baseline_lines(record: dict) -> list[str]:
    baseline = record["results"]["baseline"]
    margin = (
        f"Base model margin over chance on target benchmarks: {baseline['margin_pp']:+.1f} pp "
        f"(need {baseline['required_pp']:.1f} pp)."
    )
    return [margin, *_chance_notes(record)]


def _chance_notes(record: dict) -> list[str]:
    baseline = record["results"].get("baseline")
    if not baseline:
        return []
    lines = []
    targets = {b.name for b in get_suite(record["domain"]).by_group("target")}
    impact = record["results"].get("impact")
    if impact:
        targets &= set(impact["per_benchmark_accuracy"]["base"])
    for bench, margin in baseline.get("per_benchmark_pp", {}).items():
        if bench in targets and margin <= 0:
            lines.append(
                f"Note: {bench} is at or below chance for the base model "
                f"({margin:+.1f} pp vs. chance); gains there are unreliable."
            )
    return lines


def _warnings(record: dict) -> list[str]:
    lines = (
        _contamination_warning(record),
        _truncation_line(record),
        _eval_truncation_line(record),
    )
    return [line for line in lines if line]


def render_report(record: dict) -> str:
    """Render the full Markdown report."""
    results = record["results"]
    impact = results.get("impact")
    out = [f"# {_header(record)}", "", _context_line(record), ""]
    for note in _sample_notes(record):
        out += [note, ""]

    stop = _stop_line(record)
    if stop:
        out += [stop, ""]

    if impact:
        out += [_headline(record), ""]

    for warning in _warnings(record):
        out += [warning, ""]

    if impact:
        out += ["## Side effects", ""] + _side_effects_table(impact) + [""]
        out += ["## Per-benchmark accuracy (%)", ""] + _per_benchmark_table(impact) + [""]
        for note in _chance_notes(record):
            out += [note, ""]

    out += ["## Data checks", ""]
    if "novelty" in results:
        out += _novelty_lines(results["novelty"]) + [""]
    if "exposure" in results:
        out += _exposure_lines(results["exposure"]) + [""]
    if "duplicates" in results:
        d = results["duplicates"]
        out += [
            (
                f"Exact duplicates: {d['exact_fraction']:.1%} · "
                f"near duplicates: {d['near_fraction']:.1%}"
            ),
            "",
        ]
    if "baseline" in results:
        out += ["## Baseline", ""] + _baseline_lines(record) + [""]

    if "impact" in results:  # the caveats are about the training runs
        out += ["## Caveats", ""] + [f"- {c}" for c in CAVEATS] + [""]
    return "\n".join(out)


def render_summary(record: dict) -> str:
    """Render the short terminal summary: header, headline, side effects, warnings."""
    out = [_header(record), *_sample_notes(record)]
    headline = _headline(record)
    if headline:
        out.append(headline)
        impact = record["results"]["impact"]
        parts = [
            f"{g}: {fmt_delta(impact['delta_data_specific'][g])}" for g in GROUPS if g != "target"
        ]
        out.append("Side effects (data-specific): " + "; ".join(parts))
    stop = _stop_line(record)
    if stop:
        out.append(stop)
    out.extend(_warnings(record))
    out.extend(_chance_notes(record))
    return "\n".join(out)
