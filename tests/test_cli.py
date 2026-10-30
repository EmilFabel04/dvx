import json
from datetime import date

import pytest
from conftest import TINY, TINY_REV
from fixtures import sample_record

from dvx import cli
from dvx.cli import main
from dvx.record import RunRecord, append_to_corpus, save_record

GOOD_LINE = json.dumps(
    {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}
)


def write_data(tmp_path, text):
    path = tmp_path / "d.jsonl"
    path.write_text(text)
    return path


def test_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for name in ("scan", "run", "resume", "report", "corpus"):
        assert name in out


def test_bad_dataset_exits_2(tmp_path, capsys):
    data = write_data(tmp_path, "not json\n")
    code = main(["scan", str(data), "--model", TINY, "--home", str(tmp_path / "h")])
    err = capsys.readouterr().err
    assert code == 2
    assert "line 1" in err and err.startswith("dvx: error:")
    assert "Traceback" not in err


def test_unknown_domain_exits_2(tmp_path, capsys):
    data = write_data(tmp_path, GOOD_LINE + "\n")
    code = main(
        ["run", str(data), "--model", TINY, "--domain", "law", "--home", str(tmp_path / "h")]
    )
    err = capsys.readouterr().err
    assert code == 2
    assert "medical-qa" in err


def test_run_builds_request(tmp_path, monkeypatch, capsys):
    seen = {}

    def fake_start_run(req, **kwargs):
        seen["req"] = req
        seen["kwargs"] = kwargs
        return RunRecord(**sample_record())

    monkeypatch.setattr(cli, "start_run", fake_start_run)
    data = write_data(tmp_path, GOOD_LINE + "\n")
    code = main(
        [
            "run", str(data), "--model", TINY, "--domain", "medical-qa", "--profile", "quick",
            "--type", "medical-qa", "--date-range", "2020-01-01", "2023-12-31",
            "--unseen-after", "2024-05-01", "--device", "cpu", "--home", str(tmp_path / "h"),
        ]
    )  # fmt: skip
    req = seen["req"]
    assert code == 0
    assert req.declared_date_range == ("2020-01-01", "2023-12-31")
    assert req.declared_type == "medical-qa"
    assert req.revision is None
    assert req.unseen_after == date(2024, 5, 1)
    assert req.home == tmp_path / "h"
    assert seen["kwargs"] == {"device": "cpu"}
    assert "report.md" in capsys.readouterr().out


def test_scan_and_resume_dispatch(tmp_path, monkeypatch):
    calls = []
    record = RunRecord(**sample_record(status="scan_only"))
    monkeypatch.setattr(cli, "run_scan", lambda req, **kw: calls.append(("scan", req)) or record)
    monkeypatch.setattr(
        cli, "resume_run", lambda run_id, home, **kw: calls.append(("resume", run_id)) or record
    )
    data = write_data(tmp_path, GOOD_LINE + "\n")
    home = str(tmp_path / "h")
    assert main(["scan", str(data), "--model", TINY, "--revision", TINY_REV, "--home", home]) == 0
    assert main(["resume", "abc", "--home", home]) == 0
    assert calls[0][0] == "scan" and calls[0][1].domain == "medical-qa"
    assert calls[0][1].revision == TINY_REV
    assert calls[1] == ("resume", "abc")


def test_resume_value_error_exits_2(tmp_path, monkeypatch, capsys):
    def boom(run_id, home, **kw):
        raise ValueError("dataset changed")

    monkeypatch.setattr(cli, "resume_run", boom)
    assert main(["resume", "x", "--home", str(tmp_path)]) == 2
    assert "dvx: error: dataset changed" in capsys.readouterr().err


def test_corpus_list_and_export(tmp_path, capsys):
    rec = RunRecord(**sample_record())
    append_to_corpus(rec, tmp_path / "corpus.jsonl")
    assert main(["corpus", "list", "--home", str(tmp_path)]) == 0
    line = capsys.readouterr().out.strip()
    assert line.startswith(f"{rec.run_id}  complete  Qwen/Qwen3-0.6B  abcdef012345  ")
    assert "pp" in line
    out = tmp_path / "out.jsonl"
    assert main(["corpus", "export", str(out), "--home", str(tmp_path)]) == 0
    assert json.loads(out.read_text())["run_id"] == rec.run_id


def test_corpus_list_no_impact(tmp_path, capsys):
    rec = sample_record(status="scan_only")
    rec["results"].pop("impact")
    append_to_corpus(RunRecord(**rec), tmp_path / "corpus.jsonl")
    main(["corpus", "list", "--home", str(tmp_path)])
    assert capsys.readouterr().out.strip().endswith("  -")


def test_report_rerender(tmp_path, capsys):
    rec = RunRecord(**sample_record())
    run_dir = tmp_path / "runs" / rec.run_id
    save_record(rec, run_dir)
    (run_dir / "report.md").write_text("stale")
    assert main(["report", rec.run_id, "--home", str(tmp_path)]) == 0
    assert (run_dir / "report.md").read_text() != "stale"
    assert "report.md" in capsys.readouterr().out


def test_report_missing_run_exits_2(tmp_path, capsys):
    assert main(["report", "nope", "--home", str(tmp_path)]) == 2
    assert "nope" in capsys.readouterr().err


def test_home_defaults_to_env(monkeypatch, tmp_path):
    monkeypatch.setenv("DVX_HOME", str(tmp_path / "envhome"))
    args = cli._parser().parse_args(["resume", "x"])
    assert args.home == tmp_path / "envhome"
