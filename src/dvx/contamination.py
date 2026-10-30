"""Benchmark contamination detection via shared word n-grams."""

import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from dvx.data import Example
from dvx.evals.registry import MCQItem
from dvx.scan.duplicates import normalize_text

_PUNCT = re.compile(r"[^\w\s]")


@dataclass(frozen=True)
class ContaminationResult:
    n_flagged: int
    flagged_lines: list[int]
    by_benchmark: dict[str, int]
    ngram: int


def _words(text: str) -> list[str]:
    return _PUNCT.sub(" ", normalize_text(text)).split()


def _ngrams(words: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def find_contamination(
    examples: Sequence[Example], items: Sequence[MCQItem], n: int = 13
) -> ContaminationResult:
    """Flag examples sharing a word n-gram with an eval item (or containing a short item)."""
    index: dict[tuple[str, ...], set[str]] = defaultdict(set)
    short: list[tuple[str, str]] = []  # (benchmark, space-padded text)
    for item in items:
        words = _words(item.question + " " + " ".join(item.options))
        if not words:
            continue
        if len(words) >= n:
            for gram in _ngrams(words, n):
                index[gram].add(item.benchmark)
        else:
            short.append((item.benchmark, " " + " ".join(words) + " "))

    flagged: list[int] = []
    by_benchmark: dict[str, int] = defaultdict(int)
    for ex in examples:
        words = _words(ex.prompt + " " + ex.response)
        hits: set[str] = set()
        for gram in _ngrams(words, n):
            found = index.get(gram)
            if found:
                hits |= found
        if short:
            padded = " " + " ".join(words) + " "
            hits |= {bench for bench, text in short if text in padded}
        if hits:
            flagged.append(ex.line)
            for bench in hits:
                by_benchmark[bench] += 1
    return ContaminationResult(
        n_flagged=len(flagged),
        flagged_lines=sorted(flagged),
        by_benchmark=dict(by_benchmark),
        ngram=n,
    )
