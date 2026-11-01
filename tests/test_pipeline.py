import dataclasses
import json
import re
from datetime import date
from pathlib import Path

import pytest
from conftest import TINY, TINY_REV

from dvx.config import PROFILES
from dvx.data import DatasetFormatError, Example
from dvx.errors import DvxUserError
from dvx.evals.registry import MCQItem, get_suite
from dvx.record import read_corpus
from dvx.run import pipeline
from dvx.run.pipeline import RunRequest, resume_run, run_scan, start_run
from dvx.scan.calibration import CalibrationSet, CalibrationUnavailable, calibration_sha256

GROUPS = {"target", "adjacent", "side_effect"}
ITEMS = {
    b.name: [
        MCQItem(
            b.name,
            f"{b.name}-{i}",
            f"Which option is correct for item {i} of {b.name}?",
            ("alpha", "beta", "gamma", "delta"),
            i % 4,
        )
        for i in range(2)
    ]
    for b in get_suite("medical-qa").benchmarks
}
SHOTS = {
    name: [
        MCQItem(name, f"{name}-shot-{i}", f"Solved example {i}?", ("one", "two", "three"), i % 3)
        for i in range(5)
    ]
    for name in ITEMS
}
PROFILE = dataclasses.replace(
    PROFILES["quick"], train_cap=8, baseline_margin=-1.0, bootstrap_samples=50
)
SEEN = [f"Article {i} describes an old river valley and its mountain towns." for i in range(10)]
UNSEEN = [f"Article {i} reports a new event that happened in late {2025 + i}." for i in range(10)]


def fake_fetcher(unseen_after, n_per_side, cache_dir):
    return CalibrationSet(SEEN, UNSEEN, fetched_on="2026-10-30")


def offline_fetcher(unseen_after, n_per_side, cache_dir):
    raise CalibrationUnavailable("Wikipedia request failed: offline")


def fake_control(n, seed):
    return [
        Example(f"Control instruction {i}", f"Control answer {i}.", None, i + 1) for i in range(n)
    ]


def fail(*args, **kwargs):
    raise AssertionError("must not be called")


def write_data(path, n=12):
    rows = [
        {
            "prompt": f"Patient {i} presents with symptom {i}. What is the diagnosis?",
            "response": f"The likely diagnosis is condition {i}.",
        }
        for i in range(n)
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def make_request(tmp_path, **overrides):
    fields = {
        "data_path": write_data(tmp_path / "data.jsonl"),
        "model_id": TINY,
        "revision": TINY_REV,
        "domain": "medical-qa",
        "profile": "quick",
        "reference_id": TINY,
        "unseen_after": date(2024, 1, 1),
        "declared_type": None,
        "declared_date_range": None,
        "home": tmp_path / "home",
    }
    return RunRequest(**{**fields, **overrides})


def overrides(**kw):
    return {
        "profile_obj": PROFILE,
        "items": ITEMS,
        "shots": SHOTS,
        "calibration_fetcher": fake_fetcher,
        "control_loader": fake_control,
        "device": "cpu",
        **kw,
    }


def run_dir(req, rec):
    return req.home / "runs" / rec.run_id


def spy(monkeypatch, name):
    """Record the positional args of every call to pipeline.<name>, then call through."""
    calls = []
    real = getattr(pipeline, name)

    def wrapper(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(pipeline, name, wrapper)
    return calls


def assert_no_training_claims(report):
    for text in ("Impact measured by training on", "Quick profile:", "generic instruction data"):
        assert text not in report


def test_quick_end_to_end(tmp_path, monkeypatch):
    loads, releases = spy(monkeypatch, "load_model"), spy(monkeypatch, "release")
    req = make_request(
        tmp_path, declared_type="medical-qa", declared_date_range=("2020-01-01", "2023-12-31")
    )
    rec = start_run(req, **overrides())

    assert rec.status == "complete"
    assert set(rec.results) == {
        "duplicates",
        "contamination",
        "novelty",
        "exposure",
        "baseline",
        "impact",
    }
    impact = rec.results["impact"]
    assert set(impact["delta_data_specific"]) == GROUPS
    assert set(impact["delta_raw"]) == set(impact["delta_control"]) == GROUPS
    assert set(impact["accuracy"]) == {"base", "control", "tuned"}
    assert set(impact["per_benchmark_accuracy"]["tuned"]) == set(ITEMS)
    assert rec.results["novelty"]["status"] == "inconclusive"  # reference == target
    assert rec.results["exposure"]["n_calibration"] == len(SEEN) + len(UNSEEN)
    assert rec.dataset["declared_date_range"] == ["2020-01-01", "2023-12-31"]
    assert rec.model["revision"] == TINY_REV and rec.model["unseen_after"] == "2024-01-01"
    assert rec.model["params"] > 0
    assert set(rec.compute["steps"]) == {
        "dataset",
        "duplicates",
        "contamination",
        "scan_target",
        "scan_reference",
        "eval_base",
        "tuned_s0",
        "control_s0",
    }
    trained = [impact["train"][arm]["tokens_trained"] for arm in ("tuned", "control")]
    assert rec.compute["tokens_trained_total"] == sum(trained) > 0
    assert len(loads) == len(releases) == 5

    out = run_dir(req, rec)
    assert json.loads((out / "record.json").read_text())["status"] == "complete"
    assert (out / "report.md").read_text().startswith("# Data Impact Report")
    assert len(read_corpus(req.home / "corpus.jsonl")) == 1
    saved = json.loads((out / "request.json").read_text())
    assert saved["profile"] == json.loads(json.dumps(dataclasses.asdict(PROFILE)))
    assert not (out / "request.json.tmp").exists()


def test_evaluation_protocol_and_shots_recorded(tmp_path):
    req = make_request(tmp_path)
    rec = start_run(req, **overrides())
    ev = rec.evaluation
    assert ev["protocol"] == "plain-text 5-shot log-likelihood letter scoring"
    assert ev["max_len"] == PROFILE.scan.eval_max_len == 3072
    for b in get_suite("medical-qa").benchmarks:
        assert ev["shots"][b.name] == {
            "split": b.shot_split or b.split,
            "item_ids": [s.item_id for s in SHOTS[b.name]],
        }
    assert ev["n_truncated"] == {name: 0 for name in ITEMS}
    steps = run_dir(req, rec) / "steps"
    for step in ("eval_base", "tuned_s0", "control_s0"):
        saved = json.loads((steps / f"{step}.json").read_text())
        assert saved["n_truncated"] == ev["n_truncated"]


def test_baseline_chance_is_max_of_uniform_and_majority(tmp_path):
    rec = start_run(make_request(tmp_path), **overrides())
    # ITEMS have answers A and B: majority 50% beats 1/k = 25%.
    targets = ("medqa", "medmcqa", "pubmedqa")
    assert rec.results["baseline"]["chance_pp"] == dict.fromkeys(targets, 50.0)


def test_eval_steps_store_predicted_letters(tmp_path):
    req = make_request(tmp_path)
    rec = start_run(req, **overrides())
    for step in ("eval_base", "tuned_s0", "control_s0"):
        saved = json.loads((run_dir(req, rec) / "steps" / f"{step}.json").read_text())
        for name, items in ITEMS.items():
            letters = saved["predicted"][name]
            assert len(letters) == len(items) and set(letters) <= set("ABCD")
            assert saved["correct"][name] == [
                "ABCD".index(p) == it.answer for p, it in zip(letters, items, strict=True)
            ]


def test_eval_prompts_carry_the_shots(tmp_path, monkeypatch):
    calls = []
    real = pipeline.score_mcq

    def spy_score(model, tok, items, shots, **kwargs):
        calls.append((items[0].benchmark, list(shots), kwargs["max_len"]))
        return real(model, tok, items, shots, **kwargs)

    monkeypatch.setattr(pipeline, "score_mcq", spy_score)
    start_run(make_request(tmp_path), **overrides())
    assert len(calls) == 3 * len(ITEMS)  # base, tuned, control
    assert all(shots == SHOTS[name] and max_len == 3072 for name, shots, max_len in calls)


def test_novelty_gate_and_reliability_recorded(tmp_path):
    req = make_request(tmp_path)
    rec = run_scan(req, **overrides())
    novelty = rec.results["novelty"]
    # The tiny model is its own reference: identical losses, so the reference is never
    # below the target and the gate makes novelty inconclusive.
    assert novelty["status"] == "inconclusive"
    assert novelty["reason"].startswith("the reference loss is below the target loss on only 0.0%")
    assert novelty["reference"] == {"id": TINY, "revision": TINY_REV}
    rel = novelty["reliability"]
    assert rel["reference_below_target"] == 0.0
    assert (rel["min_reference_below_target"], rel["min_paired_discrimination"]) == (0.5, 0.6)
    assert 0.0 <= rel["paired_discrimination"] <= 1.0
    assert novelty["known_margin"] == 0.1 and novelty["noise_percentile"] == 10.0
    assert isinstance(novelty["noise_threshold"], float)
    assert novelty["loss_quartiles"]["target"] == novelty["loss_quartiles"]["reference"]
    assert len(novelty["loss_quartiles"]["target"]) == 3
    shown = [e for bucket in novelty["examples"].values() for e in bucket]
    assert shown and all(len(e["response"]) <= 80 for e in shown)
    assert all(len(b) <= 2 for b in novelty["examples"].values())
    ref = json.loads((run_dir(req, rec) / "steps" / "scan_reference.json").read_text())
    assert len(ref["mismatched_true_index"]) == len(ref["mismatched_losses"]) == 12


def test_reproducibility_fields(tmp_path):
    req = make_request(tmp_path)
    rec = start_run(req, **overrides())
    assert (rec.model["reference_id"], rec.model["reference_revision"]) == (TINY, TINY_REV)
    assert rec.results["exposure"]["calibration_set"] == {
        "source": "English Wikipedia article intros",
        "sha256": calibration_sha256(SEEN, UNSEEN),
        "fetched_on": "2026-10-30",
        "n_seen": len(SEEN),
        "n_unseen": len(UNSEEN),
    }
    env = rec.env
    assert re.fullmatch(r"[0-9a-f]{40}", env["dvx_commit"]) and isinstance(env["dvx_dirty"], bool)
    for key in ("machine", "processor", "device_name", "ram_bytes"):
        assert env[key], key
    request = json.loads((run_dir(req, rec) / "request.json").read_text())
    assert request["engine"] == {"commit": env["dvx_commit"], "dirty": env["dvx_dirty"]}
    assert rec.dataset["n_trained"] == 8


def test_unregistered_reference_without_revision_records_the_loaded_commit(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "_reference_revision", lambda req: None)
    loads = spy(monkeypatch, "load_model")
    req = make_request(tmp_path)
    rec = run_scan(req, **overrides())
    assert [rev for _, rev, _ in loads] == [TINY_REV, None]  # target pinned; reference unpinned
    assert rec.model["reference_revision"] == TINY_REV  # the HF cache snapshot it loaded
    assert rec.results["novelty"]["reference"] == {"id": TINY, "revision": TINY_REV}
    step = json.loads((run_dir(req, rec) / "steps" / "scan_reference.json").read_text())
    assert step["commit"] == TINY_REV


def test_no_calibration_set_without_calibration(tmp_path):
    rec = run_scan(make_request(tmp_path), **overrides(calibration_fetcher=offline_fetcher))
    assert rec.results["exposure"]["calibration_set"] is None


def test_resume_warns_when_the_engine_source_changed(tmp_path, monkeypatch, capsys):
    req = make_request(tmp_path)
    rec = run_scan(req, **overrides())
    (run_dir(req, rec) / "record.json").unlink()
    started = json.loads((run_dir(req, rec) / "request.json").read_text())["engine"]
    capsys.readouterr()
    other = {"commit": "f" * 40, "dirty": True}
    monkeypatch.setattr(pipeline, "engine_source", lambda: other)
    resume_run(rec.run_id, req.home, **overrides())
    err = capsys.readouterr().err
    assert f"dvx: warning: run {rec.run_id} was started with dvx commit {started['commit']}" in err
    assert f"now {'f' * 40} (dirty)" in err


def test_resume_is_quiet_when_the_engine_source_matches(tmp_path, capsys):
    req = make_request(tmp_path)
    rec = run_scan(req, **overrides())
    (run_dir(req, rec) / "record.json").unlink()
    capsys.readouterr()
    resume_run(rec.run_id, req.home, **overrides())
    assert "warning" not in capsys.readouterr().err


def test_no_absolute_paths_in_record_corpus_or_report(tmp_path):
    req = make_request(tmp_path)
    rec = start_run(req, **overrides())
    out = run_dir(req, rec)
    home = str(tmp_path)
    assert home not in json.dumps(dataclasses.asdict(rec))
    assert home not in (req.home / "corpus.jsonl").read_text()
    assert home not in (out / "report.md").read_text()
    assert rec.dataset["name"] == "data.jsonl"
    assert json.loads((out / "request.json").read_text())["request"]["data_path"].startswith(home)


def test_steps_report_progress_on_stderr(tmp_path, capsys):
    req = make_request(tmp_path)
    rec = run_scan(req, **overrides())
    err = capsys.readouterr().err.splitlines()
    for step in ("dataset", "duplicates", "contamination", "scan_target", "scan_reference"):
        assert f"dvx: step {step} started" in err
        assert any(re.fullmatch(rf"dvx: step {step} finished in \d+\.\d s", line) for line in err)
    (run_dir(req, rec) / "record.json").unlink()
    resume_run(rec.run_id, req.home, **overrides())
    assert "dvx: step scan_target cached" in capsys.readouterr().err.splitlines()


def test_baseline_stop(tmp_path):
    req = make_request(tmp_path)
    rec = start_run(
        req, **overrides(profile_obj=dataclasses.replace(PROFILE, baseline_margin=0.99))
    )
    assert rec.status == "stopped_baseline"
    assert rec.dataset["n_trained"] is None
    assert set(rec.results) == {"duplicates", "contamination", "novelty", "exposure", "baseline"}
    out = run_dir(req, rec)
    assert not (out / "steps" / "tuned_s0.json").exists()
    assert not (out / "steps" / "control_s0.json").exists()
    report = (out / "report.md").read_text()
    assert "Stopped:" in report
    assert_no_training_claims(report)
    assert len(read_corpus(req.home / "corpus.jsonl")) == 1


def test_resume_skips_completed(tmp_path, monkeypatch):
    req = make_request(tmp_path)
    rec = start_run(req, **overrides())
    out = run_dir(req, rec)
    (out / "steps" / "control_s0.json").unlink()
    (out / "record.json").unlink()
    (req.home / "corpus.jsonl").write_text("")

    finetunes, loads = spy(monkeypatch, "finetune_lora"), spy(monkeypatch, "load_model")
    resumed = resume_run(rec.run_id, req.home, **overrides())

    assert len(finetunes) == 1
    assert len(loads) == 1  # only control_s0 ran; every other step came from its file
    assert resumed.status == "complete"
    assert (resumed.run_id, resumed.created_at) == (rec.run_id, rec.created_at)
    for arm in ("base", "tuned"):  # from cached steps
        assert resumed.results["impact"]["accuracy"][arm] == rec.results["impact"]["accuracy"][arm]
    assert (out / "record.json").exists()
    assert [row["run_id"] for row in read_corpus(req.home / "corpus.jsonl")] == [rec.run_id]


def test_resume_finished_run_returns_record(tmp_path, monkeypatch):
    req = make_request(tmp_path)
    rec = run_scan(req, **overrides())
    saved = json.loads((run_dir(req, rec) / "record.json").read_text())
    for name in ("load_examples", "load_model", "load_suite_items", "load_suite_shots"):
        monkeypatch.setattr(pipeline, name, fail)

    resumed = resume_run(rec.run_id, req.home)

    assert dataclasses.asdict(resumed) == saved
    assert len(read_corpus(req.home / "corpus.jsonl")) == 1


def test_resume_interrupted_scan_stays_scan_only(tmp_path, monkeypatch):
    req = make_request(tmp_path)
    rec = run_scan(req, **overrides())
    out = run_dir(req, rec)
    (out / "record.json").unlink()  # as if stopped after the corpus append
    monkeypatch.setattr(pipeline, "load_model", fail)

    resumed = resume_run(rec.run_id, req.home, **overrides())

    assert resumed.status == "scan_only"
    assert not (out / "steps" / "eval_base.json").exists()
    assert (out / "record.json").exists()
    assert len(read_corpus(req.home / "corpus.jsonl")) == 1


def test_resume_rejects_changed_dataset(tmp_path):
    req = make_request(tmp_path)
    rec = run_scan(req, **overrides())
    (run_dir(req, rec) / "record.json").unlink()
    write_data(req.data_path, n=11)
    with pytest.raises(ValueError, match="changed"):
        resume_run(rec.run_id, req.home, **overrides())


@pytest.mark.parametrize("saved_version", ["1", None])
def test_resume_refuses_a_run_started_under_another_suite_version(tmp_path, saved_version):
    req = make_request(tmp_path)
    rec = run_scan(req, **overrides())
    out = run_dir(req, rec)
    (out / "record.json").unlink()
    request = json.loads((out / "request.json").read_text())
    assert request["suite_version"] == "2"
    if saved_version is None:
        del request["suite_version"]  # written before the field: suite v1
    else:
        request["suite_version"] = saved_version
    (out / "request.json").write_text(json.dumps(request))
    with pytest.raises(DvxUserError, match="suite v1; this dvx scores medical-qa with v2"):
        resume_run(rec.run_id, req.home, **overrides())


def test_resume_unknown_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="nope"):
        resume_run("nope", tmp_path / "home")


def test_resume_uses_stored_profile(tmp_path):
    req = make_request(tmp_path)
    rec = run_scan(req, **overrides())
    (run_dir(req, rec) / "record.json").unlink()
    other = dataclasses.replace(PROFILE, train_cap=4)
    with pytest.raises(DvxUserError, match="profile"):
        resume_run(rec.run_id, req.home, **overrides(profile_obj=other))

    resumed = resume_run(rec.run_id, req.home, **overrides(profile_obj=None))

    assert resumed.config == dataclasses.asdict(PROFILE)  # not PROFILES["quick"]


def test_relative_data_path_resumes_from_elsewhere(tmp_path, monkeypatch):
    write_data(tmp_path / "data.jsonl")
    monkeypatch.chdir(tmp_path)
    req = make_request(tmp_path, data_path=Path("data.jsonl"))
    rec = run_scan(req, **overrides())
    absolute = str((tmp_path / "data.jsonl").resolve())
    out = run_dir(req, rec)
    assert json.loads((out / "request.json").read_text())["request"]["data_path"] == absolute
    assert rec.dataset["name"] == "data.jsonl" and "path" not in rec.dataset

    (out / "record.json").unlink()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert resume_run(rec.run_id, req.home, **overrides()).status == "scan_only"


def test_unknown_model_and_offline(tmp_path, monkeypatch):
    loads = spy(monkeypatch, "load_model")
    req = make_request(tmp_path, revision=None, reference_id=None)
    rec = start_run(req, **overrides(calibration_fetcher=offline_fetcher))
    assert rec.status == "complete"
    novelty, exposure = rec.results["novelty"], rec.results["exposure"]
    assert novelty["status"] == "unavailable" and novelty["reason"] == "no reference model"
    assert exposure["status"] == "inconclusive"
    assert exposure["reason"].startswith("calibration unavailable")
    assert rec.model["family"] == "unknown" and rec.model["reference_id"] is None
    assert not (run_dir(req, rec) / "steps" / "scan_reference.json").exists()
    # No revision requested: the record pins the commit scan_target loaded,
    # and every later step loads exactly that commit.
    commit = rec.model["revision"]
    assert re.fullmatch(r"[0-9a-f]{40}", commit)
    assert [rev for _, rev, _ in loads] == [None, commit, commit, commit]


def test_scan_only(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "load_suite_shots", fail)  # a scan needs no shots
    req = make_request(tmp_path, unseen_after=None)
    rec = run_scan(req, **overrides(calibration_fetcher=fail, shots=None))
    assert rec.status == "scan_only"
    assert rec.evaluation is None
    assert set(rec.results) == {"duplicates", "contamination", "novelty", "exposure"}
    assert rec.results["novelty"]["status"] == "inconclusive"
    exposure = rec.results["exposure"]
    assert exposure["status"] == "inconclusive"
    assert exposure["reason"] == "model release date unknown; pass --unseen-after"
    assert rec.compute["tokens_trained_total"] == 0
    out = run_dir(req, rec)
    assert not (out / "steps" / "eval_base.json").exists()
    assert_no_training_claims((out / "report.md").read_text())  # 8 of 12 would be "trained"
    assert len(read_corpus(req.home / "corpus.jsonl")) == 1


def test_contamination_checks_the_whole_scan_sample(tmp_path, monkeypatch):
    checked = []
    real = pipeline.find_contamination

    def spy_check(examples, items):
        checked.append(len(examples))
        return real(examples, items)

    monkeypatch.setattr(pipeline, "find_contamination", spy_check)
    profile = dataclasses.replace(PROFILE, measure_cap=10)  # train_cap 8
    rec = run_scan(make_request(tmp_path), **overrides(profile_obj=profile))
    assert checked == [rec.dataset["n_scanned"]] == [10]


def test_sample_counts(tmp_path):
    req = make_request(tmp_path)
    rec = run_scan(req, **overrides(profile_obj=dataclasses.replace(PROFILE, measure_cap=10)))
    assert rec.dataset["n_total"] == 12
    assert rec.dataset["n_scanned"] == 10
    assert rec.dataset["n_trained"] is None  # a scan trains nothing
    assert rec.results["duplicates"]["n"] == 12
    assert sum(rec.results["novelty"]["counts"].values()) == 10


def test_bad_dataset_fails_before_loading(tmp_path, monkeypatch):
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json\n")
    for name in ("load_suite_items", "load_model"):
        monkeypatch.setattr(pipeline, name, fail)
    for entry in (start_run, run_scan):
        req = make_request(tmp_path, data_path=bad)
        with pytest.raises(DatasetFormatError, match="line 1"):
            entry(req, calibration_fetcher=fail, control_loader=fail)
    assert not (tmp_path / "home").exists()


def test_saved_profile_without_gradient_checkpointing_loads_with_default():
    saved = dataclasses.asdict(PROFILES["quick"])
    del saved["train"]["gradient_checkpointing"]  # request.json written before the field
    profile = pipeline._profile_from_json(json.loads(json.dumps(saved)))
    assert profile == PROFILES["quick"] and profile.train.gradient_checkpointing is True
