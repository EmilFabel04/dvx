"""Dataset loading, sampling, fingerprinting and control data."""

import hashlib
import json
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

DOLLY_ID = "databricks/databricks-dolly-15k"
DOLLY_REVISION = "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"

_MISSING = "missing 'prompt'/'response' or 'messages'"
_MULTI_TURN = "multi-turn conversations are not supported in v1"


@dataclass(frozen=True)
class Example:
    prompt: str
    response: str
    system: str | None
    line: int


class DatasetFormatError(ValueError):
    pass


def _parse_messages(messages: object, n: int) -> tuple[str | None, object, object]:
    if not isinstance(messages, list) or not all(
        isinstance(m, dict) and "role" in m and "content" in m for m in messages
    ):
        raise DatasetFormatError(f"line {n}: {_MISSING}")
    roles = [m["role"] for m in messages]
    system = None
    body = messages
    if roles and roles[0] == "system":
        system = messages[0]["content"]
        body = messages[1:]
        roles = roles[1:]
    if roles == ["user", "assistant"]:
        return system, body[0]["content"], body[1]["content"]
    if roles.count("user") > 1 or roles.count("assistant") > 1 or "system" in roles:
        raise DatasetFormatError(f"line {n}: {_MULTI_TURN}")
    raise DatasetFormatError(f"line {n}: {_MISSING}")


def _parse_line(text: str, n: int) -> Example:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DatasetFormatError(f"line {n}: invalid JSON ({exc.msg})") from exc
    if not isinstance(obj, dict):
        raise DatasetFormatError(f"line {n}: {_MISSING}")
    if "prompt" in obj and "response" in obj:
        system, prompt, response = obj.get("system"), obj["prompt"], obj["response"]
    elif "messages" in obj:
        system, prompt, response = _parse_messages(obj["messages"], n)
    else:
        raise DatasetFormatError(f"line {n}: {_MISSING}")
    if not isinstance(prompt, str) or not isinstance(response, str):
        raise DatasetFormatError(f"line {n}: prompt and response must be strings")
    if system is not None and not isinstance(system, str):
        raise DatasetFormatError(f"line {n}: system must be strings")
    if not response.strip():
        raise DatasetFormatError(f"line {n}: empty response")
    return Example(prompt=prompt, response=response, system=system, line=n)


def load_examples(path: Path) -> list[Example]:
    examples = []
    with open(path, encoding="utf-8") as f:
        for n, raw in enumerate(f, start=1):
            if raw.strip():
                examples.append(_parse_line(raw, n))
    if not examples:
        raise DatasetFormatError(f"no examples found in {path}")
    return examples


def fingerprint(examples: list[Example]) -> str:
    h = hashlib.sha256()
    for e in examples:
        row = json.dumps(
            [e.system, e.prompt, e.response],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        h.update(row.encode("utf-8") + b"\n")
    return h.hexdigest()


def sample_examples(examples: list[Example], cap: int, seed: int) -> list[Example]:
    if len(examples) <= cap:
        return examples
    picked = random.Random(seed).sample(examples, cap)
    return sorted(picked, key=lambda e: e.line)


def load_control_examples(n: int, seed: int, rows: Iterable[dict] | None = None) -> list[Example]:
    if rows is None:
        from datasets import load_dataset

        rows = load_dataset(DOLLY_ID, split="train", revision=DOLLY_REVISION)
    examples = []
    for i, row in enumerate(rows, start=1):
        prompt = row["instruction"]
        if row.get("context"):
            prompt += "\n\n" + row["context"]
        examples.append(Example(prompt=prompt, response=row["response"], system=None, line=i))
    if n > len(examples):
        raise ValueError(f"requested {n} control examples but only {len(examples)} available")
    return sample_examples(examples, n, seed) if n < len(examples) else examples
