"""Per-domain registry of pinned multiple-choice evaluation suites."""

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from dvx.errors import DvxUserError
from dvx.scan.duplicates import normalize_text

Group = Literal["target", "adjacent", "side_effect"]

LETTERS = "ABCDEFGH"  # option labels; no registered benchmark has more than 5 options
N_SHOTS = 5
SHOT_SEED = 0


@dataclass(frozen=True)
class MCQItem:
    """One multiple-choice question. The number of options varies (ARC has 3-5)."""

    benchmark: str
    item_id: str
    question: str
    options: tuple[str, ...]
    answer: int


@dataclass(frozen=True)
class Benchmark:
    """A pinned Hub benchmark.

    `size` is the maximum number of items; some splits are smaller
    (e.g. MMLU college_biology has 144). `shot_split` is the split of the same pinned
    dataset that few-shot examples come from; None means the eval split's own rows that
    were not sampled as eval items (for datasets with a single split).
    """

    name: str
    group: Group
    hf_id: str
    config: str | None
    split: str
    revision: str
    size: int
    to_item: Callable[[dict, int], MCQItem]
    shot_split: str | None


@dataclass(frozen=True)
class Suite:
    domain: str
    version: str
    benchmarks: tuple[Benchmark, ...]

    def by_group(self, group: Group) -> list[Benchmark]:
        return [b for b in self.benchmarks if b.group == group]


SuiteItems = dict[str, list[MCQItem]]


class UnknownDomain(DvxUserError):
    pass


def _medqa(row: dict, idx: int) -> MCQItem:
    options = tuple(row["options"][k] for k in "ABCD")
    return MCQItem(
        "medqa", f"medqa-{idx}", row["question"], options, "ABCD".index(row["answer_idx"])
    )


def _medmcqa(row: dict, idx: int) -> MCQItem:
    options = (row["opa"], row["opb"], row["opc"], row["opd"])
    return MCQItem("medmcqa", row["id"], row["question"], options, int(row["cop"]))


_PUBMED_OPTIONS = ("yes", "no", "maybe")


def _pubmedqa(row: dict, idx: int) -> MCQItem:
    question = (
        "Context: " + " ".join(row["context"]["contexts"]) + "\n\nQuestion: " + row["question"]
    )
    return MCQItem(
        "pubmedqa",
        f"pubmedqa-{row['pubid']}",
        question,
        _PUBMED_OPTIONS,
        _PUBMED_OPTIONS.index(row["final_decision"]),
    )


def _mmlu(name: str) -> Callable[[dict, int], MCQItem]:
    def convert(row: dict, idx: int) -> MCQItem:
        return MCQItem(
            name,
            f"mmlu-{row['subject']}-{idx}",
            row["question"],
            tuple(row["choices"]),
            int(row["answer"]),
        )

    return convert


def _arc(row: dict, idx: int) -> MCQItem:
    choices = row["choices"]
    return MCQItem(
        "arc_challenge",
        row["id"],
        row["question"],
        tuple(choices["text"]),
        choices["label"].index(row["answerKey"]),
    )


_MMLU_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"


def _mmlu_bench(subject: str, group: Group) -> Benchmark:
    name = f"mmlu_{subject}"
    return Benchmark(
        name, group, "cais/mmlu", subject, "test", _MMLU_REVISION, 200, _mmlu(name), "dev"
    )


# v2: plain-text 5-shot letter scoring, shots drawn per benchmark with SHOT_SEED.
_MEDICAL_QA = Suite(
    domain="medical-qa",
    version="2",
    benchmarks=(
        Benchmark(
            "medqa",
            "target",
            "GBaker/MedQA-USMLE-4-options",
            None,
            "test",
            "0fb93dd23a7339b6dcd27e241cb9b5eca62d4d18",
            500,
            _medqa,
            "train",
        ),
        Benchmark(
            "medmcqa",
            "target",
            "openlifescienceai/medmcqa",
            None,
            "validation",
            "91c6572c454088bf71b679ad90aa8dffcd0d5868",
            500,
            _medmcqa,
            "train",
        ),
        Benchmark(
            "pubmedqa",
            "target",
            "qiaojin/PubMedQA",
            "pqa_labeled",
            "train",
            "9001f2853fb87cab8d220904e0de81ac6973b318",
            500,
            _pubmedqa,
            None,  # pqa_labeled has only a train split
        ),
        _mmlu_bench("college_biology", "adjacent"),
        _mmlu_bench("high_school_biology", "adjacent"),
        _mmlu_bench("college_chemistry", "adjacent"),
        _mmlu_bench("high_school_chemistry", "adjacent"),
        Benchmark(
            "arc_challenge",
            "side_effect",
            "allenai/ai2_arc",
            "ARC-Challenge",
            "test",
            "210d026faf9955653af8916fad021475a3f00453",
            500,
            _arc,
            "train",
        ),
        _mmlu_bench("high_school_mathematics", "side_effect"),
        _mmlu_bench("philosophy", "side_effect"),
        _mmlu_bench("high_school_macroeconomics", "side_effect"),
        _mmlu_bench("miscellaneous", "side_effect"),
    ),
)

SUITES: dict[str, Suite] = {_MEDICAL_QA.domain: _MEDICAL_QA}


def get_suite(domain: str) -> Suite:
    try:
        return SUITES[domain]
    except KeyError:
        raise UnknownDomain(
            f"unknown domain {domain!r}; available: {', '.join(sorted(SUITES))}"
        ) from None


def _hub_rows(bench: Benchmark, split: str):
    from datasets import load_dataset

    return load_dataset(bench.hf_id, bench.config, split=split, revision=bench.revision)


def load_items(bench: Benchmark, rows: list[dict] | None = None, seed: int = 0) -> list[MCQItem]:
    """Convert benchmark rows to items, sampling `bench.size` of them if there are more."""
    if rows is None:
        rows = _hub_rows(bench, bench.split)
    items = sorted((bench.to_item(row, i) for i, row in enumerate(rows)), key=lambda x: x.item_id)
    if len(items) > bench.size:
        items = sorted(random.Random(seed).sample(items, bench.size), key=lambda x: x.item_id)
    return items


def load_shots(
    bench: Benchmark,
    items: list[MCQItem],
    rows: list[dict] | None = None,
    seed: int = SHOT_SEED,
    n: int = N_SHOTS,
) -> list[MCQItem]:
    """`n` few-shot examples for `bench`, in prompt order, disjoint from its eval `items`.

    Rows are visited in a `random.Random(seed)` permutation of `bench.shot_split`'s rows
    (or, for None, the eval split's). A row is skipped if it is one of the eval items or
    restates an eval item's question (after normalize_text), so no shot gives away an
    eval answer.
    """
    if rows is None:
        rows = _hub_rows(bench, bench.shot_split or bench.split)
    eval_ids = {it.item_id for it in items} if bench.shot_split is None else set()
    eval_questions = {normalize_text(it.question) for it in items}
    shots: list[MCQItem] = []
    for i in random.Random(seed).sample(range(len(rows)), len(rows)):
        item = bench.to_item(rows[i], i)
        if item.item_id in eval_ids or normalize_text(item.question) in eval_questions:
            continue
        shots.append(item)
        if len(shots) == n:
            return shots
    raise ValueError(
        f"{bench.name}: only {len(shots)} rows disjoint from the eval items; need {n} shots"
    )


def load_suite_items(suite: Suite) -> SuiteItems:
    return {b.name: load_items(b) for b in suite.benchmarks}


def load_suite_shots(suite: Suite, items: SuiteItems) -> SuiteItems:
    return {b.name: load_shots(b, items[b.name]) for b in suite.benchmarks}
