"""Exact and near-duplicate detection."""

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from datasketch import MinHash, MinHashLSH

from dvx.data import Example

_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class DuplicateStats:
    n: int
    exact_dup: int
    near_dup: int
    exact_fraction: float
    near_fraction: float


def normalize_text(s: str) -> str:
    """NFKC, lowercase, collapse whitespace to single spaces, strip."""
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", s).lower()).strip()


def _minhash(text: str, shingle_words: int, num_perm: int) -> MinHash | None:
    words = text.split()
    if not words:
        return None
    if len(words) < shingle_words:
        shingles = {" ".join(words)}
    else:
        shingles = {
            " ".join(words[i : i + shingle_words])
            for i in range(len(words) - shingle_words + 1)
        }
    mh = MinHash(num_perm=num_perm, seed=1)
    for sh in shingles:
        mh.update(sh.encode("utf-8"))
    return mh


def duplicate_stats(
    examples: Sequence[Example],
    threshold: float = 0.8,
    shingle_words: int = 5,
    num_perm: int = 128,
) -> DuplicateStats:
    seen: set[str] = set()
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    exact = near = 0
    for i, example in enumerate(examples):
        text = normalize_text(example.prompt + "\n" + example.response)
        if text in seen:
            exact += 1
            continue
        seen.add(text)
        mh = _minhash(text, shingle_words, num_perm)
        if mh is None:
            continue
        if lsh.query(mh):
            near += 1
        lsh.insert(str(i), mh)
    n = len(examples)
    return DuplicateStats(
        n=n,
        exact_dup=exact,
        near_dup=near,
        exact_fraction=exact / n if n else 0.0,
        near_fraction=near / n if n else 0.0,
    )
