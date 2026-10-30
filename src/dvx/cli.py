"""Command-line interface: scan, run, resume, report and corpus subcommands."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

from dvx.config import PROFILES
from dvx.data import DatasetFormatError
from dvx.evals.registry import UnknownDomain, get_suite
from dvx.record import RunRecord, read_corpus
from dvx.report import fmt_delta, render_report, render_summary
from dvx.run.pipeline import RunRequest, resume_run, run_scan, start_run

# ValueError covers a changed dataset on resume, a profile mismatch and a corrupt corpus.
USER_ERRORS = (DatasetFormatError, UnknownDomain, FileNotFoundError, ValueError, FileExistsError)


def _home_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--home",
        type=Path,
        default=Path(os.environ.get("DVX_HOME") or ".dvx"),
        help="working directory for runs and the corpus (default: $DVX_HOME or .dvx)",
    )


def _request_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("data", type=Path, help="dataset: a JSONL file of chat examples")
    p.add_argument("--model", required=True, help="model to measure (Hub id or local folder)")
    p.add_argument("--revision", help="model revision (default: the registry's pinned one)")
    p.add_argument("--reference", help="reference model for the novelty check")
    p.add_argument(
        "--unseen-after",
        type=date.fromisoformat,
        metavar="YYYY-MM-DD",
        help="the model's release date, for the exposure check",
    )
    _home_arg(p)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dvx", description="Measure what a text dataset does to an LLM."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="check a dataset against a model, without training")
    _request_args(scan)
    scan.add_argument("--domain", default="medical-qa", help="evaluation domain")

    run = sub.add_parser("run", help="scan a dataset and measure its impact by fine-tuning")
    _request_args(run)
    run.add_argument("--domain", required=True, help="evaluation domain")
    run.add_argument("--profile", choices=sorted(PROFILES), default="quick")
    run.add_argument("--type", dest="declared_type", help="declared dataset type")
    run.add_argument(
        "--date-range", nargs=2, metavar=("START", "END"), help="declared dataset date range"
    )
    run.add_argument("--device", choices=["cuda", "mps", "cpu"], help="default: best available")

    resume = sub.add_parser("resume", help="continue an interrupted run")
    resume.add_argument("run_id")
    _home_arg(resume)

    report = sub.add_parser("report", help="re-render a finished run's report")
    report.add_argument("run_id")
    _home_arg(report)

    corpus = sub.add_parser("corpus", help="inspect the corpus of finished runs")
    corpus_sub = corpus.add_subparsers(dest="corpus_command", required=True)
    corpus_list = corpus_sub.add_parser("list", help="one line per run")
    _home_arg(corpus_list)
    export = corpus_sub.add_parser("export", help="write the corpus as JSONL")
    export.add_argument("out", type=Path)
    _home_arg(export)
    return parser


def _request(args: argparse.Namespace, profile: str) -> RunRequest:
    get_suite(args.domain)  # fail on an unknown domain before any loading
    date_range = getattr(args, "date_range", None)
    return RunRequest(
        data_path=args.data,
        model_id=args.model,
        revision=args.revision,
        domain=args.domain,
        profile=profile,
        reference_id=args.reference,
        unseen_after=args.unseen_after,
        declared_type=getattr(args, "declared_type", None),
        declared_date_range=(date_range[0], date_range[1]) if date_range else None,
        home=args.home,
    )


def _print_run(record: RunRecord, home: Path) -> None:
    print(render_summary(asdict(record)))
    print(f"Report: {home / 'runs' / record.run_id / 'report.md'}")


def _report(args: argparse.Namespace) -> None:
    run_dir = args.home / "runs" / args.run_id
    path = run_dir / "record.json"
    if not path.exists():
        raise FileNotFoundError(f"no finished run {args.run_id!r} in {args.home / 'runs'}")
    record = json.loads(path.read_text(encoding="utf-8"))
    (run_dir / "report.md").write_text(render_report(record), encoding="utf-8")
    print(render_summary(record))
    print(f"Report: {run_dir / 'report.md'}")


def _headline(row: dict) -> str:
    impact = row["results"].get("impact")
    return fmt_delta(impact["delta_data_specific"]["target"]) if impact else "-"


def _corpus(args: argparse.Namespace) -> None:
    rows = read_corpus(args.home / "corpus.jsonl")
    if args.corpus_command == "list":
        for row in rows:
            fp = row["dataset"]["fingerprint"][:12]
            print(f"{row['run_id']}  {row['status']}  {row['model']['id']}  {fp}  {_headline(row)}")
    else:
        args.out.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
        )


def _dispatch(args: argparse.Namespace) -> None:
    if args.command == "scan":
        _print_run(run_scan(_request(args, "quick")), args.home)
    elif args.command == "run":
        record = start_run(_request(args, args.profile), device=args.device)
        _print_run(record, args.home)
    elif args.command == "resume":
        _print_run(resume_run(args.run_id, args.home), args.home)
    elif args.command == "report":
        _report(args)
    else:
        _corpus(args)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _dispatch(args)
    except USER_ERRORS as err:
        print(f"dvx: error: {err}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
