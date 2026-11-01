"""Run record: one JSON document per run, plus an append-only corpus of runs."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import dvx
from dvx.errors import DvxUserError

SCHEMA_VERSION = 2
_PACKAGE_DIR = Path(dvx.__file__).resolve().parent


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
    evaluation: dict | None = None  # added in schema 2; None for a scan


def new_run_id(now: datetime, fingerprint: str) -> str:
    return f"{now:%Y%m%d-%H%M%S}-{fingerprint[:8]}"


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(_PACKAGE_DIR), *args],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    ).stdout.strip()


def engine_source() -> dict:
    """The git commit of the dvx source tree and whether it has uncommitted changes.

    Both None when dvx does not run from a git checkout of itself (for example, an
    installed wheel, possibly inside some other repository) or git is unavailable.
    """
    try:
        top = Path(_git("rev-parse", "--show-toplevel"))
        if (top / "src" / "dvx").resolve() != _PACKAGE_DIR:
            return {"commit": None, "dirty": None}
        return {"commit": _git("rev-parse", "HEAD"), "dirty": bool(_git("status", "--porcelain"))}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def _chip() -> str | None:
    """The CPU or SoC name, e.g. "Apple M5"; None if it can't be read."""
    try:
        if sys.platform == "darwin":
            cmd = ["sysctl", "-n", "machdep.cpu.brand_string"]
            out = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=10)
            return out.stdout.strip() or None
        if sys.platform.startswith("linux"):
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or None


def _ram_bytes() -> int | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def _device_name(device: str, chip: str | None) -> str | None:
    """The CUDA device name; for MPS and CPU, the chip (torch names no MPS device)."""
    if device == "cuda":
        import torch

        return torch.cuda.get_device_name(0)
    return chip


def env_info(device: str) -> dict:
    """Library versions, the dvx source commit, and the hardware the run used."""
    source = engine_source()
    chip = _chip()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": version("torch"),
        "transformers": version("transformers"),
        "peft": version("peft"),
        "datasets": version("datasets"),
        "device": device,
        "dvx": dvx.__version__,
        "dvx_commit": source["commit"],
        "dvx_dirty": source["dirty"],
        "machine": platform.machine(),
        "processor": chip,
        "device_name": _device_name(device, chip),
        "ram_bytes": _ram_bytes(),
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
            raise DvxUserError(f"{corpus_path}: line {n} is not valid JSON ({exc.msg})") from exc
    return rows
