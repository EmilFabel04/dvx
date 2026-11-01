"""Step-cached pipeline: dataset scan, quick impact run, and resume.

A run lives in ``home/runs/<run_id>/``: ``request.json`` (what was asked), one JSON
file per finished step under ``steps/``, then ``report.md`` and ``record.json``.
Each step loads the models it needs and releases them before returning, so at most
one model is in memory at a time.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import get_args

import numpy as np
from huggingface_hub import try_to_load_from_cache
from transformers import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

import dvx
from dvx.config import PROFILES, Profile, ScanConfig, TrainConfig
from dvx.contamination import find_contamination
from dvx.data import Example, fingerprint, load_control_examples, load_examples, sample_examples
from dvx.errors import DvxUserError
from dvx.evals.registry import (
    LETTERS,
    N_SHOTS,
    Group,
    Suite,
    SuiteItems,
    get_suite,
    load_suite_items,
    load_suite_shots,
)
from dvx.evals.scorer import PROTOCOL, score_mcq
from dvx.evals.stats import (
    BaselineCheck,
    baseline_check,
    chance_level,
    group_accuracy,
    paired_delta,
)
from dvx.models import count_params, load_model, model_info, pick_device, release
from dvx.record import (
    SCHEMA_VERSION,
    RunRecord,
    append_to_corpus,
    engine_source,
    env_info,
    new_run_id,
    read_corpus,
    save_record,
)
from dvx.report import render_report
from dvx.run.steps import cached_step, write_json_atomic
from dvx.scan.calibration import (
    CALIBRATION_SOURCE,
    CalibrationSet,
    CalibrationUnavailable,
    calibration_sha256,
    fetch_calibration_texts,
)
from dvx.scan.duplicates import duplicate_stats
from dvx.scan.exposure import assess_exposure, exposure_inconclusive, min_k_score
from dvx.scan.losses import response_losses, token_logprobs
from dvx.scan.novelty import (
    NoveltyResult,
    bucket_examples,
    classify,
    mismatched_pairs,
    novelty_unavailable,
)
from dvx.train import finetune_lora

GROUPS: tuple[Group, ...] = get_args(Group)
SEED = 0  # scan and train samples, mismatched pairs, control sample, bootstrap
NO_RELEASE_DATE = "model release date unknown; pass --unseen-after"
NO_REFERENCE = "no reference model"
TOO_FEW_EXAMPLES = "need at least 2 examples"
_SNAPSHOT = re.compile(r"snapshots/([0-9a-f]{40})/")

CalibrationFetcher = Callable[[date, int, Path], CalibrationSet]
ControlLoader = Callable[[int, int], list[Example]]


@dataclass(frozen=True)
class RunRequest:
    data_path: Path
    model_id: str
    revision: str | None
    domain: str
    profile: str
    reference_id: str | None
    unseen_after: date | None
    declared_type: str | None
    declared_date_range: tuple[str, str] | None
    home: Path


@dataclass
class _Run:
    """One run's resolved inputs, plus the outputs of the steps it has passed."""

    req: RunRequest
    run_id: str
    created_at: str
    scan_only: bool
    examples: list[Example]
    fingerprint: str
    profile: Profile
    suite: Suite
    items: SuiteItems
    shots: SuiteItems  # few-shot examples per benchmark; empty for a scan
    device: str
    calibration_fetcher: CalibrationFetcher
    control_loader: ControlLoader
    outputs: dict[str, dict] = field(default_factory=dict)
    reference_revision: str | None = None  # resolved when the reference step runs

    @property
    def dir(self) -> Path:
        return self.req.home / "runs" / self.run_id

    @property
    def seed(self) -> int:
        return self.profile.seeds[0]

    def step(self, name: str, fn: Callable[[], dict]) -> dict:
        self.outputs[name] = cached_step(self.dir, name, fn)
        return self.outputs[name]


def start_run(
    req: RunRequest,
    *,
    profile_obj: Profile | None = None,
    items: SuiteItems | None = None,
    shots: SuiteItems | None = None,
    calibration_fetcher: CalibrationFetcher = fetch_calibration_texts,
    control_loader: ControlLoader = load_control_examples,
    device: str | None = None,
    now: datetime | None = None,
) -> RunRecord:
    """Scan the dataset, then measure its impact with one tuned and one control fine-tune."""
    return _new_run(
        req,
        scan_only=False,
        now=now,
        profile_obj=profile_obj,
        items=items,
        shots=shots,
        calibration_fetcher=calibration_fetcher,
        control_loader=control_loader,
        device=device,
    )


def run_scan(
    req: RunRequest,
    *,
    profile_obj: Profile | None = None,
    items: SuiteItems | None = None,
    shots: SuiteItems | None = None,
    calibration_fetcher: CalibrationFetcher = fetch_calibration_texts,
    control_loader: ControlLoader = load_control_examples,
    device: str | None = None,
    now: datetime | None = None,
) -> RunRecord:
    """Run only the scan steps (no evaluation or training); the status is "scan_only"."""
    return _new_run(
        req,
        scan_only=True,
        now=now,
        profile_obj=profile_obj,
        items=items,
        shots=shots,
        calibration_fetcher=calibration_fetcher,
        control_loader=control_loader,
        device=device,
    )


def resume_run(
    run_id: str,
    home: Path,
    *,
    profile_obj: Profile | None = None,
    items: SuiteItems | None = None,
    shots: SuiteItems | None = None,
    calibration_fetcher: CalibrationFetcher = fetch_calibration_texts,
    control_loader: ControlLoader = load_control_examples,
    device: str | None = None,
) -> RunRecord:
    """Continue a run from its saved steps. A finished run's record is returned as saved."""
    home = Path(home)
    run_dir = home / "runs" / run_id
    request_path = run_dir / "request.json"
    if not request_path.exists():
        raise FileNotFoundError(f"no run {run_id!r} in {home / 'runs'}")
    record_path = run_dir / "record.json"
    if record_path.exists():
        return RunRecord(**json.loads(record_path.read_text(encoding="utf-8")))
    saved = json.loads(request_path.read_text(encoding="utf-8"))
    _warn_if_engine_changed(run_id, saved.get("engine", {"commit": None, "dirty": None}))
    profile = _profile_from_json(saved["profile"])
    if profile_obj is not None and profile_obj != profile:
        raise DvxUserError(
            f"the profile override differs from the profile run {run_id} was started with; "
            "resume without one so cached and new steps share a config"
        )
    req = replace(_request_from_json(saved["request"]), home=home)
    examples = load_examples(req.data_path)
    run = _prepare(
        req,
        run_id,
        saved["created_at"],
        saved["scan_only"],
        examples,
        profile_obj=profile,
        items=items,
        shots=shots,
        calibration_fetcher=calibration_fetcher,
        control_loader=control_loader,
        device=device,
    )
    return _execute(run)


def _new_run(req: RunRequest, *, scan_only: bool, now: datetime | None, **resolve) -> RunRecord:
    examples = load_examples(req.data_path)  # first, so format errors fail before any loading
    now = now or datetime.now(UTC)
    run_id = new_run_id(now, fingerprint(examples))
    run = _prepare(_resolve_request(req), run_id, now.isoformat(), scan_only, examples, **resolve)
    run.dir.mkdir(parents=True)  # FileExistsError rather than reuse another run's steps
    request = {
        "created_at": run.created_at,
        "scan_only": scan_only,
        "profile": asdict(run.profile),
        "request": _request_to_json(run.req),
        "engine": engine_source(),
    }
    write_json_atomic(run.dir / "request.json", request, indent=2)
    return _execute(run)


def _describe(source: dict) -> str:
    commit = source.get("commit") or "unknown"
    return commit + (" (dirty)" if source.get("dirty") else "")


def _warn_if_engine_changed(run_id: str, started: dict) -> None:
    now = engine_source()
    if started != now:
        print(
            f"dvx: warning: run {run_id} was started with dvx commit {_describe(started)}, "
            f"now {_describe(now)}; its cached steps came from the earlier code.",
            file=sys.stderr,
        )


def _prepare(
    req: RunRequest,
    run_id: str,
    created_at: str,
    scan_only: bool,
    examples: list[Example],
    *,
    profile_obj: Profile | None,
    items: SuiteItems | None,
    shots: SuiteItems | None,
    calibration_fetcher: CalibrationFetcher,
    control_loader: ControlLoader,
    device: str | None,
) -> _Run:
    profile = profile_obj or PROFILES[req.profile]
    if len(profile.seeds) != 1:
        raise ValueError(
            f"the pipeline trains exactly one seed; profile {profile.name!r} has {profile.seeds}"
        )
    suite = get_suite(req.domain)
    if items is None:
        items = load_suite_items(suite)
    if shots is None:
        shots = {} if scan_only else load_suite_shots(suite, items)
    return _Run(
        req=req,
        run_id=run_id,
        created_at=created_at,
        scan_only=scan_only,
        examples=examples,
        fingerprint=fingerprint(examples),
        profile=profile,
        suite=suite,
        items=items,
        shots=shots,
        device=device or pick_device(),
        calibration_fetcher=calibration_fetcher,
        control_loader=control_loader,
    )


def _resolve_request(req: RunRequest) -> RunRequest:
    """Fill model defaults from the registry and make the data path absolute, so a run
    can be resumed from any working directory."""
    info = model_info(req.model_id)
    return replace(
        req,
        data_path=Path(req.data_path).resolve(),
        revision=req.revision or info.revision,
        reference_id=req.reference_id or info.reference_id,
        unseen_after=req.unseen_after or info.unseen_after,
    )


def _reference_revision(req: RunRequest) -> str | None:
    """The reference's pinned revision: the target's if they are the same model, else the
    registry's; None if neither pins one (the reference step then records the commit)."""
    if req.reference_id is None:
        return None
    if req.reference_id == req.model_id:
        return req.revision
    return model_info(req.reference_id).revision


def _request_to_json(req: RunRequest) -> dict:
    return {
        **asdict(req),
        "data_path": str(req.data_path),
        "home": str(req.home),
        "unseen_after": req.unseen_after.isoformat() if req.unseen_after else None,
        "declared_date_range": _date_range(req),
    }


def _request_from_json(d: dict) -> RunRequest:
    return RunRequest(
        **{
            **d,
            "data_path": Path(d["data_path"]),
            "home": Path(d["home"]),
            "unseen_after": date.fromisoformat(d["unseen_after"]) if d["unseen_after"] else None,
            "declared_date_range": (
                tuple(d["declared_date_range"]) if d["declared_date_range"] else None
            ),
        }
    )


def _profile_from_json(d: dict) -> Profile:
    train = d["train"]
    return Profile(
        **{
            **d,
            "seeds": tuple(d["seeds"]),
            "train": TrainConfig(
                **{**train, "lora_target_modules": tuple(train["lora_target_modules"])}
            ),
            "scan": ScanConfig(**d["scan"]),
        }
    )


def _date_range(req: RunRequest) -> list[str] | None:
    return list(req.declared_date_range) if req.declared_date_range else None


def _execute(run: _Run) -> RunRecord:
    ds = run.step("dataset", lambda: _dataset_step(run))
    if ds["fingerprint"] != run.fingerprint:
        raise DvxUserError(
            f"dataset {run.req.data_path} changed since run {run.run_id} started "
            f"(fingerprint was {ds['fingerprint'][:12]}, now {run.fingerprint[:12]})"
        )
    by_line = {e.line: e for e in run.examples}
    scan = [by_line[n] for n in ds["scan_lines"]]
    train = [by_line[n] for n in ds["train_lines"]]
    results = {
        "duplicates": _public(run.step("duplicates", lambda: _duplicates_step(run))),
        "contamination": _public(
            run.step("contamination", lambda: _contamination_step(run, scan))
        ),
    }
    target = run.step("scan_target", lambda: _scan_target_step(run, scan))
    if run.req.revision is None:  # pin later loads and the record to the commit just loaded
        run.req = replace(run.req, revision=target["commit"])
    run.reference_revision = _reference_revision(run.req)
    skip = _reference_skip(run.req, scan)
    if skip:
        novelty = novelty_unavailable(skip, len(scan))
    else:
        ref = run.step("scan_reference", lambda: _scan_reference_step(run, scan))
        run.reference_revision = run.reference_revision or ref["commit"]
        novelty = _novelty(run, target, ref)
    results["novelty"] = _novelty_summary(run, novelty, scan)
    results["exposure"] = _exposure(run, target)
    if run.scan_only:
        return _finish(run, "scan_only", results, ds, target)

    base = run.step("eval_base", lambda: _eval_base_step(run))
    check = _baseline(run, base["correct"])
    results["baseline"] = asdict(check)
    if not check.ok:
        return _finish(run, "stopped_baseline", results, ds, target)

    tuned = run.step(f"tuned_s{run.seed}", lambda: _finetune_step(run, train))
    control = run.step(
        f"control_s{run.seed}",
        lambda: _finetune_step(run, run.control_loader(len(train), SEED)),
    )
    results["impact"] = _impact(run, base, tuned, control)
    return _finish(run, "complete", results, ds, target)


# --- steps ----------------------------------------------------------------------------


@contextmanager
def _loaded(
    model_id: str, revision: str | None, device: str
) -> Iterator[tuple[PreTrainedModel, PreTrainedTokenizerBase]]:
    """A freshly loaded model, released on exit (it is unusable afterwards)."""
    model, tokenizer = load_model(model_id, revision, device)
    try:
        yield model, tokenizer
    finally:
        release(model)


def _dataset_step(run: _Run) -> dict:
    scan = sample_examples(run.examples, run.profile.measure_cap, SEED)
    train = sample_examples(scan, run.profile.train_cap, SEED)
    return {
        "fingerprint": run.fingerprint,
        "n_total": len(run.examples),
        "scan_lines": [e.line for e in scan],
        "train_lines": [e.line for e in train],
    }


def _duplicates_step(run: _Run) -> dict:
    cfg = run.profile.scan
    stats = duplicate_stats(run.examples, cfg.minhash_threshold, cfg.shingle_words, cfg.num_perm)
    return asdict(stats)


def _contamination_step(run: _Run, scan: list[Example]) -> dict:
    """Checks the whole scan sample (up to measure_cap), which contains the train sample."""
    items = [item for b in run.suite.benchmarks for item in run.items[b.name]]
    return asdict(find_contamination(scan, items))


def _scan_target_step(run: _Run, scan: list[Example]) -> dict:
    cal, reason = _calibration_texts(run)
    seen, unseen = (cal.seen, cal.unseen) if cal else ([], [])
    texts = [e.prompt + "\n" + e.response for e in scan]
    with _loaded(run.req.model_id, run.req.revision, run.device) as (model, tok):
        return {
            "n_params": count_params(model),
            "commit": _loaded_commit(run.req.model_id, run.req.revision),
            "losses": _losses(run, model, tok, scan),
            "scores": _min_k_scores(run, model, tok, texts),
            "calibration": {
                "reason": reason,
                "seen": _min_k_scores(run, model, tok, seen),
                "unseen": _min_k_scores(run, model, tok, unseen),
                "set": _calibration_set(cal),
            },
        }


def _calibration_set(cal: CalibrationSet | None) -> dict | None:
    """What the calibration texts were: their hash, fetch date and counts."""
    if cal is None:
        return None
    return {
        "source": CALIBRATION_SOURCE,
        "sha256": calibration_sha256(cal.seen, cal.unseen),
        "fetched_on": cal.fetched_on,
        "n_seen": len(cal.seen),
        "n_unseen": len(cal.unseen),
    }


def _scan_reference_step(run: _Run, scan: list[Example]) -> dict:
    mismatched, true_index = mismatched_pairs(scan, run.profile.scan.mismatched_n, seed=SEED)
    ref_id, revision = run.req.reference_id, _reference_revision(run.req)
    with _loaded(ref_id, revision, run.device) as (model, tok):
        return {
            "commit": _loaded_commit(ref_id, revision),
            "losses": _losses(run, model, tok, scan),
            "mismatched_losses": _losses(run, model, tok, mismatched),
            "mismatched_true_index": true_index,  # scan index of each pair's true pair
        }


def _eval_base_step(run: _Run) -> dict:
    with _loaded(run.req.model_id, run.req.revision, run.device) as (model, tok):
        return _evaluate(run, model, tok)


def _finetune_step(run: _Run, examples: list[Example]) -> dict:
    # A fresh target every time: peft wraps the model in place.
    with _loaded(run.req.model_id, run.req.revision, run.device) as (model, tok):
        tuned, stats = finetune_lora(model, tok, examples, run.profile.train, seed=run.seed)
        return {"train": asdict(stats), **_evaluate(run, tuned, tok)}


def _loaded_commit(model_id: str, revision: str | None) -> str | None:
    """The commit a just-loaded Hub model came from: the HF cache snapshot folder of its
    config.json. None for a local folder or when the cache can't tell.

    (transformers 5 no longer sets `config._commit_hash`; the cache layout is the same
    across versions.)
    """
    try:
        path = try_to_load_from_cache(model_id, "config.json", revision=revision)
    except ValueError:  # not a Hub repo id, e.g. a local folder
        return None
    match = _SNAPSHOT.search(Path(path).as_posix()) if isinstance(path, str) else None
    return match.group(1) if match else None


def _calibration_texts(run: _Run) -> tuple[CalibrationSet | None, str | None]:
    """(texts, reason); the reason says why there are no calibration texts."""
    if run.req.unseen_after is None:
        return None, NO_RELEASE_DATE
    cfg = run.profile.scan
    try:
        cal = run.calibration_fetcher(
            run.req.unseen_after, cfg.calibration_per_side, run.req.home / "calibration"
        )
    except CalibrationUnavailable as err:
        return None, f"calibration unavailable: {err}"
    return cal, None


def _losses(run: _Run, model, tok, examples: list[Example]) -> list[float | None]:
    """Response losses; a non-finite one is stored as JSON null (classify excludes it)."""
    losses = response_losses(
        model,
        tok,
        examples,
        max_len=run.profile.train.max_seq_len,
        batch_size=run.profile.scan.batch_size,
    )
    return [v if np.isfinite(v) else None for v in losses]


def _min_k_scores(run: _Run, model, tok, texts: list[str]) -> list[float | None]:
    """Min-k% scores. A text too short to score (NaN) is stored as JSON null; numpy
    reads None back as NaN, which assess_exposure drops."""
    cfg = run.profile.scan
    logprobs = token_logprobs(
        model, tok, texts, max_len=cfg.score_max_len, batch_size=cfg.batch_size
    )
    scores = [min_k_score(lp, k=cfg.min_k) for lp in logprobs]
    return [None if np.isnan(s) else s for s in scores]


def _evaluate(run: _Run, model, tok) -> dict:
    """Per benchmark: each item's predicted letter and correctness, and the number of
    left-truncated prompts."""
    cfg = run.profile.scan
    scores = {
        b.name: score_mcq(
            model,
            tok,
            run.items[b.name],
            run.shots[b.name],
            batch_size=cfg.batch_size,
            max_len=cfg.eval_max_len,
        )
        for b in run.suite.benchmarks
    }
    return {
        "predicted": {name: [LETTERS[p] for p in s.predicted] for name, s in scores.items()},
        "correct": {name: s.correct.tolist() for name, s in scores.items()},
        "n_truncated": {name: s.n_truncated for name, s in scores.items()},
    }


# --- assembly -------------------------------------------------------------------------


def _public(out: dict) -> dict:
    return {k: v for k, v in out.items() if k != "_wall_seconds"}


def _reference_skip(req: RunRequest, scan: list[Example]) -> str | None:
    if req.reference_id is None:
        return NO_REFERENCE
    if len(scan) < 2:
        return TOO_FEW_EXAMPLES
    return None


def _novelty(run: _Run, target: dict, ref: dict) -> NoveltyResult:
    cfg = run.profile.scan
    return classify(
        target["losses"],
        ref["losses"],
        ref["mismatched_losses"],
        [ref["losses"][j] for j in ref["mismatched_true_index"]],
        cfg.known_margin,
        cfg.noise_percentile,
        cfg.novelty_min_reference_below_target,
        cfg.novelty_min_paired_discrimination,
    )


def _novelty_summary(run: _Run, novelty: NoveltyResult, scan: list[Example]) -> dict:
    """The novelty record: the result, the reference that produced it, example items."""
    ref = run.req.reference_id
    return {
        **novelty.summary(),
        "reference": None if ref is None else {"id": ref, "revision": run.reference_revision},
        "examples": bucket_examples(novelty.labels, scan),
    }


def _exposure(run: _Run, target: dict) -> dict:
    cal = target["calibration"]
    if cal["reason"]:
        result = exposure_inconclusive(cal["reason"])
    else:
        result = assess_exposure(
            target["scores"], cal["seen"], cal["unseen"], min_auc=run.profile.scan.exposure_min_auc
        )
    return {**asdict(result), "calibration_set": cal.get("set")}


def _arrays(correct: dict[str, list[bool]], names: list[str]) -> dict[str, np.ndarray]:
    return {n: np.asarray(correct[n], dtype=bool) for n in names}


def _baseline(run: _Run, correct: dict[str, list[bool]]) -> BaselineCheck:
    names = [b.name for b in run.suite.by_group("target")]
    chance = {}
    for name in names:
        items = run.items[name]
        k = Counter(len(item.options) for item in items).most_common(1)[0][0]
        chance[name] = chance_level([item.answer for item in items], k)
    return baseline_check(_arrays(correct, names), chance, run.profile.baseline_margin)


def _impact(run: _Run, base: dict, tuned: dict, control: dict) -> dict:
    n_boot = run.profile.bootstrap_samples
    arms = {"base": base["correct"], "control": control["correct"], "tuned": tuned["correct"]}
    names = {g: [b.name for b in run.suite.by_group(g)] for g in GROUPS}

    def group(arm: str, g: Group) -> dict[str, np.ndarray]:
        return _arrays(arms[arm], names[g])

    def delta(before: str, after: str) -> dict:
        return {
            g: asdict(paired_delta(group(before, g), group(after, g), n_boot, SEED)) for g in GROUPS
        }

    return {
        "accuracy": {
            arm: {g: asdict(group_accuracy(group(arm, g), n_boot, SEED)) for g in GROUPS}
            for arm in arms
        },
        "per_benchmark_accuracy": {
            arm: {n: float(np.mean(c[n]) * 100.0) for n in c} for arm, c in arms.items()
        },
        "delta_data_specific": delta("control", "tuned"),
        "delta_raw": delta("base", "tuned"),
        "delta_control": delta("base", "control"),
        "train": {"tuned": tuned["train"], "control": control["train"]},
    }


def _evaluation(run: _Run) -> dict | None:
    """How the eval steps scored: protocol, shot item IDs and truncation; None for a scan."""
    base = run.outputs.get("eval_base")
    if base is None:
        return None
    return {
        "protocol": PROTOCOL,
        "n_shots": N_SHOTS,
        "max_len": run.profile.scan.eval_max_len,
        "shots": {
            b.name: {
                "split": b.shot_split or b.split,
                "item_ids": [s.item_id for s in run.shots[b.name]],
            }
            for b in run.suite.benchmarks
        },
        "n_truncated": base["n_truncated"],
    }


def _compute(run: _Run) -> dict:
    trained = [out["train"] for out in run.outputs.values() if "train" in out]
    return {
        "steps": {
            name: {"wall_seconds": out["_wall_seconds"]} for name, out in run.outputs.items()
        },
        "tokens_trained_total": sum(t["tokens_trained"] for t in trained),
        "est_flops_total": sum(t["est_flops"] for t in trained),
    }


def _finish(run: _Run, status: str, results: dict, ds: dict, target: dict) -> RunRecord:
    req = run.req
    record = RunRecord(
        schema_version=SCHEMA_VERSION,
        run_id=run.run_id,
        created_at=run.created_at,
        status=status,
        engine_version=dvx.__version__,
        profile=run.profile.name,
        domain=req.domain,
        suite_version=run.suite.version,
        env=env_info(run.device),
        dataset={
            "name": req.data_path.name,  # no absolute path: request.json keeps that
            "fingerprint": ds["fingerprint"],
            "n_total": ds["n_total"],
            "n_scanned": len(ds["scan_lines"]),
            "n_trained": len(ds["train_lines"]) if status == "complete" else None,
            "declared_type": req.declared_type,
            "declared_date_range": _date_range(req),
        },
        model={
            "id": req.model_id,
            "revision": req.revision,
            "params": target["n_params"],
            "family": model_info(req.model_id).family,
            "reference_id": req.reference_id,
            "reference_revision": run.reference_revision,
            "unseen_after": req.unseen_after.isoformat() if req.unseen_after else None,
        },
        seeds=list(run.profile.seeds),
        config=asdict(run.profile),
        results=results,
        compute=_compute(run),
        evaluation=_evaluation(run),
    )
    (run.dir / "report.md").write_text(render_report(asdict(record)), encoding="utf-8")
    corpus = req.home / "corpus.jsonl"
    # A resumed run may have stopped after this append but before record.json was saved.
    if all(row.get("run_id") != run.run_id for row in read_corpus(corpus)):
        append_to_corpus(record, corpus)
    save_record(record, run.dir)  # last: record.json marks the run finished
    return record
