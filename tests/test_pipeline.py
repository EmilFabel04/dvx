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
from dvx.scan.calibration import CalibrationUnavailable

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
PROFILE = dataclasses.replace(
    PROFILES["quick"], train_cap=8, baseline_margin=-1.0, bootstrap_samples=50
)
SEEN = [f"Article {i} describes an old river valley and its mountain towns." for i in range(10)]
UNSEEN = [f"Article {i} reports a new event that happened in late {2025 + i}." for i in range(10)]


def fake_fetcher(unseen_after, n_per_side, cache_dir):
    return SEEN, UNSEEN


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
    assert rec.results["novelty"]["status"] == "ok"
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


def test_baseline_stop(tmp_path):
    req = make_request(tmp_path)
    rec = start_run(
        req, **overrides(profile_obj=dataclasses.replace(PROFILE, baseline_margin=0.99))
    )
    assert rec.status == "stopped_baseline"
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
    for name in ("load_examples", "load_model", "load_suite_items"):
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
    assert rec.dataset["path"] == absolute

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


def test_scan_only(tmp_path):
    req = make_request(tmp_path, unseen_after=None)
    rec = run_scan(req, **overrides(calibration_fetcher=fail))
    assert rec.status == "scan_only"
    assert set(rec.results) == {"duplicates", "contamination", "novelty", "exposure"}
    assert rec.results["novelty"]["status"] == "ok"
    exposure = rec.results["exposure"]
    assert exposure["status"] == "inconclusive"
    assert exposure["reason"] == "model release date unknown; pass --unseen-after"
    assert rec.compute["tokens_trained_total"] == 0
    out = run_dir(req, rec)
    assert not (out / "steps" / "eval_base.json").exists()
    assert_no_training_claims((out / "report.md").read_text())  # 8 of 12 would be "trained"
    assert len(read_corpus(req.home / "corpus.jsonl")) == 1


def test_sample_counts(tmp_path):
    req = make_request(tmp_path)
    rec = run_scan(req, **overrides(profile_obj=dataclasses.replace(PROFILE, measure_cap=10)))
    assert rec.dataset["n_total"] == 12
    assert rec.dataset["n_scanned"] == 10
    assert rec.dataset["n_trained"] == 8
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
