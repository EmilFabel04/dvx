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
    with corpus_path.open("a") as f:
        f.write(line + "\n")


def read_corpus(corpus_path: Path) -> list[dict]:
    corpus_path = Path(corpus_path)
    if not corpus_path.exists():
        return []
    with corpus_path.open() as f:
        return [json.loads(line) for line in f if line.strip()]
