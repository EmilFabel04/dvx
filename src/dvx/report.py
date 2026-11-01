"""Render a run record (dict) as a Markdown report and a terminal summary.

Pure functions: dict in, str out. The pipeline and CLI write the files.
"""

from pathlib import PurePath

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


def _count(n: int, singular: str, plural: str | None = None) -> str:
    """'1 example', '3 examples' (thousands separated)."""
    return f"{n:,} {singular if n == 1 else plural or singular + 's'}"


def _ordinal(x: float) -> str:
    n = round(x)
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _ci(e: dict) -> str:
    return f"95% CI {e['ci_low']:+.1f} to {e['ci_high']:+.1f}"


def fmt_delta(e: dict) -> str:
    """Format an Estimate of a delta, in percentage points with its 95% CI; a CI that
    includes zero, or lies wholly below it, is labelled."""
    text = f"{e['value']:+.1f} pp ({_ci(e)})"
    if e["ci_low"] <= 0 <= e["ci_high"]:
        text += " — not distinguishable from zero"
    elif e["ci_high"] < 0:
        text += " — regression"
    return text


def _header(record: dict) -> str:
    d = record["dataset"]
    name = d.get("name") or PurePath(d["path"]).name  # schema 1 recorded the full path
    model = record["model"]["id"]
    return f"Data Impact Report: {model} on {name} (fingerprint {d['fingerprint'][:12]})"


def _headline(record: dict) -> str | None:
    impact = record["results"].get("impact")
    if not impact:
        return None
    return "Data-specific gain on target benchmarks: " + fmt_delta(
        impact["delta_data_specific"]["target"]
    )


def _seeds(record: dict) -> str:
    n = len(record["seeds"])
    return "one training seed" if n == 1 else f"{n} training seeds"


def _regression_lines(record: dict) -> list[str]:
    """One line per group whose data-specific delta is wholly below zero."""
    impact = record["results"].get("impact")
    if not impact:
        return []
    lines = []
    for g in GROUPS:
        e = impact["delta_data_specific"][g]
        if e["ci_high"] < 0:
            lines.append(
                f"Regression vs. control: {g} {e['value']:+.1f} pp ({_ci(e)}; {_seeds(record)}). "
                f"Versus the untuned base: {impact['delta_raw'][g]['value']:+.1f} pp."
            )
    return lines


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
    verb = "overlaps" if c["n_flagged"] == 1 else "overlap"
    return (
        f"WARNING: {_count(c['n_flagged'], 'example')} {verb} with evaluation items ({by}). "
        "Gains on those benchmarks may be inflated."
    )


def _truncation_line(record: dict) -> str | None:
    impact = record["results"].get("impact")
    if not impact:
        return None
    tuned, control = (impact["train"][arm]["n_truncated"] for arm in ("tuned", "control"))
    if tuned <= 0 and control <= 0:
        return None
    max_len = record["config"]["train"]["max_seq_len"]
    return (
        f"Training examples truncated to {max_len} tokens: tuned {tuned:,}, control {control:,}."
    )


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
    total = _count(d["n_total"], "example")
    if d["n_scanned"] < d["n_total"]:
        notes.append(f"Measured on a sample of {d['n_scanned']:,} of {total}.")
    if "impact" in record["results"] and d["n_trained"] < d["n_total"]:
        notes.append(f"Impact measured by training on {d['n_trained']:,} of {total}.")
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
    header = (
        "| Group | Data-specific (tuned − control) | Raw (tuned − base) "
        "| Format effect (control − base) |"
    )
    lines = [header, "|---|---|---|---|"]
    for g in GROUPS:
        cells = (impact[key][g] for key in ("delta_data_specific", "delta_raw", "delta_control"))
        lines.append(f"| {g} | " + " | ".join(fmt_delta(e) for e in cells) + " |")
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
    """Paragraphs: the bucket mix only when the gate passed, else the reason; either way the
    reliability, reference, thresholds, loss quartiles and example items."""
    if novelty["status"] == "unavailable":
        return [f"Novelty: unavailable ({novelty['reason']})"]
    ok = novelty["status"] == "ok"
    if ok:
        f = novelty["fractions"]
        out = [
            (
                f"Known to the model: {f['known']:.1%} · learnable: {f['learnable']:.1%} "
                f"· noise: {f['noise']:.1%}"
            )
        ]
    else:
        out = [f"Novelty: inconclusive ({novelty['reason']})."]
    rel = novelty.get("reliability")
    if rel:
        out.append(
            "Reliability: the reference loss is below the target loss on "
            f"{rel['reference_below_target']:.1%} of examples "
            f"(gate: at least {rel['min_reference_below_target']:.0%}); "
            f"paired discrimination {rel['paired_discrimination']:.2f} "
            f"(gate: at least {rel['min_paired_discrimination']:.2f})."
        )
    ref = novelty.get("reference")
    if ref and novelty.get("noise_threshold") is not None:
        out.append(
            f"Reference model: {ref['id']} (revision {ref['revision']}). Thresholds: known "
            f"margin {novelty['known_margin']:.2f} nats per token; noise threshold "
            f"{novelty['noise_threshold']:.2f} nats per token (the "
            f"{_ordinal(novelty['noise_percentile'])} percentile of mismatched-pair reference "
            "loss)."
        )
    q = novelty.get("loss_quartiles")
    if q:
        t, r = (" / ".join(f"{v:.2f}" for v in q[k]) for k in ("target", "reference"))
        out.append(f"Loss quartiles (nats per response token): target {t}; reference {r}.")
    n_excluded = novelty.get("n_excluded")
    if n_excluded:
        verb = "was" if n_excluded == 1 else "were"
        out.append(
            f"{_count(n_excluded, 'example')} {verb} excluded from novelty (non-finite loss)."
        )
    shown = [(b, e) for b, items in (novelty.get("examples") or {}).items() for e in items]
    if shown:
        out.append(
            "Example items:" if ok else "Example items, as the unreliable rule buckets them:"
        )
        out.append(
            "\n".join(f'- {b}, line {e["line"]}: "{e["prompt"]}" → "{e["response"]}"'
                      for b, e in shown)
        )
    return out


def _exposure_lines(exposure: dict) -> list[str]:
    if exposure["status"] != "ok":
        return [f"Prior exposure: inconclusive ({exposure['reason']})."]
    return [
        (
            f"About {exposure['seen_fraction']:.0%} of examples look previously seen by this model "
            f"(calibration AUC {exposure['auc']:.2f})."
        ),
        "Calibration compares old vs. new Wikipedia articles; topic drift can inflate the AUC.",
        (
            "The seen/unseen threshold is calibrated on Wikipedia text and applied here to "
            "another domain; treat the fraction as indicative."
        ),
    ]


def _duplicates_line(d: dict) -> str:
    exact, near = d["exact_fraction"], d["near_fraction"]
    return f"Exact duplicates: {exact:.1%} · near duplicates: {near:.1%}"


def _baseline_lines(record: dict) -> list[str]:
    """Paragraphs of the Baseline section. The chance notes go here only when there is no
    per-benchmark table to put them under."""
    baseline = record["results"]["baseline"]
    out = [
        (
            f"Base model margin over chance on target benchmarks: {baseline['margin_pp']:+.1f} pp "
            f"(need {baseline['required_pp']:.1f} pp)."
        )
    ]
    chance = baseline.get("chance_pp")
    if chance:
        out.append(
            "Chance is the larger of 1/k and the share of the most common answer among the "
            "eval items: " + ", ".join(f"{b} {c:.1f}%" for b, c in chance.items()) + "."
        )
    if "impact" not in record["results"]:
        out += _chance_notes(record)
    return out


def _chance_notes(record: dict) -> list[str]:
    baseline = record["results"].get("baseline")
    if not baseline:
        return []
    lines = []
    targets = {b.name for b in get_suite(record["domain"]).by_group("target")}
    impact = record["results"].get("impact")
    if impact:
        targets &= set(impact["per_benchmark_accuracy"]["base"])
    chance = baseline.get("chance_pp", {})
    for bench, margin in baseline.get("per_benchmark_pp", {}).items():
        if bench in targets and margin <= 0:
            level = f" {chance[bench]:.1f}%" if bench in chance else ""
            lines.append(
                f"Note: {bench} is at or below chance for the base model "
                f"({margin:+.1f} pp vs. chance{level}); gains there are unreliable."
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
        for line in _regression_lines(record):
            out += [line, ""]

    for warning in _warnings(record):
        out += [warning, ""]

    if impact:
        out += ["## Side effects", ""] + _side_effects_table(impact) + [""]
        out += ["## Per-benchmark accuracy (%)", ""] + _per_benchmark_table(impact) + [""]
        for note in _chance_notes(record):
            out += [note, ""]

    out += ["## Data checks", ""]
    if "novelty" in results:
        for paragraph in _novelty_lines(results["novelty"]):
            out += [paragraph, ""]
    if "exposure" in results:
        out += _exposure_lines(results["exposure"]) + [""]
    if "duplicates" in results:
        out += [_duplicates_line(results["duplicates"]), ""]
    if "baseline" in results:
        out += ["## Baseline", ""]
        for paragraph in _baseline_lines(record):
            out += [paragraph, ""]

    if "impact" in results:  # the caveats are about the training runs
        out += ["## Caveats", ""] + [f"- {c}" for c in CAVEATS] + [""]
    return "\n".join(out)


def _summary_checks(results: dict) -> list[str]:
    """One line each for novelty, exposure and duplicates (or why they are inconclusive)."""
    out = []
    novelty = results.get("novelty")
    if novelty and novelty["status"] == "ok":
        f, rel = novelty["fractions"], novelty.get("reliability")
        line = (
            f"Novelty: known {f['known']:.1%} · learnable {f['learnable']:.1%} "
            f"· noise {f['noise']:.1%}"
        )
        if rel:
            line += (
                f" (reference below target on {rel['reference_below_target']:.1%} of examples; "
                f"paired discrimination {rel['paired_discrimination']:.2f})"
            )
        out.append(line)
    elif novelty:
        out.append(f"Novelty: {novelty['status']} ({novelty['reason']})")
    exposure = results.get("exposure")
    if exposure and exposure["status"] == "ok":
        out.append(
            f"Prior exposure: about {exposure['seen_fraction']:.0%} of examples look previously "
            f"seen (calibration AUC {exposure['auc']:.2f})"
        )
    elif exposure:
        out.append(f"Prior exposure: inconclusive ({exposure['reason']})")
    if "duplicates" in results:
        out.append(_duplicates_line(results["duplicates"]))
    return out


def render_summary(record: dict) -> str:
    """Render the short terminal summary: header, headline, side effects, warnings, and
    one line per data check."""
    out = [_header(record), *_sample_notes(record)]
    headline = _headline(record)
    if headline:
        out.append(headline)
        out.extend(_regression_lines(record))
        impact = record["results"]["impact"]
        out.append("Side effects (data-specific, tuned − control):")
        out += [
            f"  {g}: {fmt_delta(impact['delta_data_specific'][g])}" for g in GROUPS if g != "target"
        ]
    stop = _stop_line(record)
    if stop:
        out.append(stop)
    out.extend(_warnings(record))
    out.extend(_chance_notes(record))
    out.extend(_summary_checks(record["results"]))
    return "\n".join(out)
