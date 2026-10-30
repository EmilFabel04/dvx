"""Step cache: each pipeline step's JSON output, written atomically under steps/."""

import json
import os
import time
from collections.abc import Callable
from pathlib import Path


def cached_step(run_dir: Path, name: str, fn: Callable[[], dict]) -> dict:
    """Return the saved output of step `name`, or run `fn` and save its output.

    Only a complete `steps/{name}.json` counts: output is written to a `.tmp` file
    and moved into place with `os.replace`, so an interrupted write is never read.
    The saved output gains `"_wall_seconds"`, the time `fn` took.
    """
    steps = Path(run_dir) / "steps"
    path = steps / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    start = time.perf_counter()
    out = dict(fn())
    out["_wall_seconds"] = time.perf_counter() - start
    steps.mkdir(parents=True, exist_ok=True)
    tmp = steps / f"{name}.json.tmp"
    tmp.write_text(json.dumps(out, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)
    return out
