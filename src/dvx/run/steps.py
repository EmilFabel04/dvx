"""Step cache: each pipeline step's JSON output, written atomically under steps/."""

import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path


def write_json_atomic(path: Path, obj: object, *, indent: int | None = None) -> None:
    """Write `obj` as strict JSON via a `.tmp` file and `os.replace`: never half-written."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=indent, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def cached_step(run_dir: Path, name: str, fn: Callable[[], dict]) -> dict:
    """Return the saved output of step `name`, or run `fn` and save its output.

    Only a complete `steps/{name}.json` counts: output is written to a `.tmp` file
    and moved into place with `os.replace`, so an interrupted write is never read.
    The saved output gains `"_wall_seconds"`, the time `fn` took. One progress line
    goes to stderr when a step starts, finishes (with its wall time) or is cached.
    """
    steps = Path(run_dir) / "steps"
    path = steps / f"{name}.json"
    if path.exists():
        _progress(f"step {name} cached")
        return json.loads(path.read_text(encoding="utf-8"))
    _progress(f"step {name} started")
    start = time.perf_counter()
    out = dict(fn())
    out["_wall_seconds"] = time.perf_counter() - start
    steps.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, out)
    _progress(f"step {name} finished in {out['_wall_seconds']:.1f} s")
    return out


def _progress(message: str) -> None:
    print(f"dvx: {message}", file=sys.stderr, flush=True)
