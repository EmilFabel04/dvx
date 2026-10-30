"""Run record: one JSON document per run, plus an append-only corpus of runs."""

from __future__ import annotations

import json
import os
import platform
from dataclasses import asdict, dataclass
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import dvx

SCHEMA_VERSION = 1


@dataclass
class RunRecord:
    schema_version: int
    run_id: str
    created_at: str
    status: Literal["complete", "stopped_baseline", "scan_only"]
    engine_version: str
    profile: str
    domain: str
    suite_version: str
    env: dict
    dataset: dict
    model: dict
    seeds: list[int]
    config: dict
    results: dict
    compute: dict


def new_run_id(now: datetime, fingerprint: str) -> str:
    return f"{now:%Y%m%d-%H%M%S}-{fingerprint[:8]}"


def env_info(device: str) -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": version("torch"),
        "transformers": version("transformers"),
        "peft": version("peft"),
        "datasets": version("datasets"),
        "device": device,
        "dvx": dvx.__version__,
    }


def save_record(record: RunRecord, run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    text = json.dumps(asdict(record), indent=2, sort_keys=True, allow_nan=False)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "record.json"
    tmp = run_dir / "record.json.tmp"
    tmp.write_text(text + "\n")
    os.replace(tmp, path)
    return path


def append_to_corpus(record: RunRecord, corpus_path: Path) -> None:
    corpus_path = Path(corpus_path)
    line = json.dumps(asdict(record), sort_keys=True, allow_nan=False)
    if any(row.get("run_id") == record.run_id for row in read_corpus(corpus_path)):
        raise ValueError(f"run_id already in corpus: {record.run_id}")
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    _end_after_last_record(corpus_path)
    with corpus_path.open("a") as f:
        f.write(line + "\n")


def _parses(raw: bytes) -> bool:
    try:
        json.loads(raw)
    except ValueError:
        return False
    return True


def _end_after_last_record(corpus_path: Path) -> None:
    """Make the next append start on a fresh line after the last complete record.

    A malformed final line is an append torn by a crash (read_corpus skips it). It is
    truncated away, so it can never end up between two records.
    """
    if not corpus_path.exists():
        return
    data = corpus_path.read_bytes()
    body = data.rstrip()
    start = body.rfind(b"\n") + 1  # start of the last non-blank line
    if body and not _parses(body[start:]):
        with corpus_path.open("r+b") as f:
            f.truncate(start)
    elif data and not data.endswith(b"\n"):
        with corpus_path.open("ab") as f:
            f.write(b"\n")


def read_corpus(corpus_path: Path) -> list[dict]:
    """All records in the corpus, in order.

    A malformed final line (an append torn by a crash) is skipped; a malformed line
    anywhere else raises ValueError.
    """
    corpus_path = Path(corpus_path)
    if not corpus_path.exists():
        return []
    with corpus_path.open() as f:
        lines = [(n, line) for n, line in enumerate(f, start=1) if line.strip()]
    rows = []
    for i, (n, line) in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            if i == len(lines) - 1:
                break
            raise ValueError(f"{corpus_path}: line {n} is not valid JSON ({exc.msg})") from exc
    return rows
