"""Benchmark contamination detection via shared word n-grams and question phrases."""

import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from dvx.data import Example
from dvx.evals.registry import MCQItem
from dvx.scan.duplicates import normalize_text

_PUNCT = re.compile(r"[^\w\s]")
# An option label such as "A.", "(B)" or "c)" (letters A-H), standing alone between spaces.
_OPTION_LABEL = re.compile(r"(?<!\S)(?:\([a-h]\)|[a-h][.)])(?!\S)")
MIN_PHRASE_WORDS = 6
# Share of an item's n-grams an example must contain. Stock exam phrasing ("which of the
# following is the most appropriate next step in the management of this patient") covers
# at most 20% of a MedQA item's 13-grams; a copied item covers all of them.
MIN_COVERAGE = 0.5


@dataclass(frozen=True)
class ContaminationResult:
    n_flagged: int
    flagged_lines: list[int]
    by_benchmark: dict[str, int]
    ngram: int


def _words(text: str) -> list[str]:
    """Normalized words, with option labels and punctuation removed."""
    return _PUNCT.sub(" ", _OPTION_LABEL.sub(" ", normalize_text(text))).split()


def _ngrams(words: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def _padded(words: list[str]) -> str:
    return " " + " ".join(words) + " "


def find_contamination(
    examples: Sequence[Example], items: Sequence[MCQItem], n: int = 13
) -> ContaminationResult:
    """Flag examples that contain an eval item.

    Text is normalized (normalize_text), option labels like "A.", "(B)" and "C)" are
    dropped, and punctuation is removed. An example contains an item if it has:
      - at least MIN_COVERAGE of the word n-grams of the item's question + options, or
        of its question alone (a question of n words or more); or
      - the question as a phrase, if it has MIN_PHRASE_WORDS to n - 1 words, together
        with at least one of the item's options (so a generic question such as "Which of
        the following statements is true?" does not match on its own); or
      - for a question under MIN_PHRASE_WORDS words, question + options as a phrase, if
        those have MIN_PHRASE_WORDS to n - 1 words.
    Items with fewer words in total are too short to tell apart from ordinary text and
    are never matched.
    """
    grams: dict[tuple[str, ...], list[tuple[int, bool]]] = defaultdict(list)  # (item, in question)
    n_grams: list[tuple[int, int]] = []  # per item: (question + options, question) n-gram counts
    # Phrases keyed by their first MIN_PHRASE_WORDS words: (item, phrase, options to find).
    phrases: dict[tuple[str, ...], list[tuple[int, tuple[str, ...], list[str]]]] = defaultdict(list)
    for k, item in enumerate(items):
        question = _words(item.question)
        options = [w for w in (_words(o) for o in item.options) if w]
        whole = question + [w for o in options for w in o]
        q_grams = _ngrams(question, n)
        w_grams = _ngrams(whole, n)
        for gram in w_grams:
            grams[gram].append((k, gram in q_grams))
        n_grams.append((len(w_grams), len(q_grams)))
        if MIN_PHRASE_WORDS <= len(question) < n:
            phrase, needed = tuple(question), [_padded(o) for o in options]
        elif len(question) < MIN_PHRASE_WORDS <= len(whole) < n:
            phrase, needed = tuple(whole), []
        else:
            continue
        phrases[phrase[:MIN_PHRASE_WORDS]].append((k, phrase, needed))

    flagged: list[int] = []
    by_benchmark: dict[str, int] = defaultdict(int)
    for ex in examples:
        words = _words(ex.prompt + " " + ex.response)
        whole_hits: Counter[int] = Counter()
        question_hits: Counter[int] = Counter()
        for gram in _ngrams(words, n):
            for k, in_question in grams.get(gram, ()):
                whole_hits[k] += 1
                question_hits[k] += in_question
        found = {k for k, c in whole_hits.items() if c >= MIN_COVERAGE * n_grams[k][0]}
        found |= {k for k, c in question_hits.items() if c and c >= MIN_COVERAGE * n_grams[k][1]}
        padded = _padded(words)
        for i in range(len(words) - MIN_PHRASE_WORDS + 1):
            for k, phrase, needed in phrases.get(tuple(words[i : i + MIN_PHRASE_WORDS]), ()):
                if tuple(words[i : i + len(phrase)]) == phrase and (
                    not needed or any(option in padded for option in needed)
                ):
                    found.add(k)
        hits = {items[k].benchmark for k in found}
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
