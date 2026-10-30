"""Per-domain registry of pinned multiple-choice evaluation suites."""

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from dvx.errors import DvxUserError

Group = Literal["target", "adjacent", "side_effect"]


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
    (e.g. MMLU college_biology has 144).
    """

    name: str
    group: Group
    hf_id: str
    config: str | None
    split: str
    revision: str
    size: int
    to_item: Callable[[dict, int], MCQItem]


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
    return Benchmark(name, group, "cais/mmlu", subject, "test", _MMLU_REVISION, 200, _mmlu(name))


_MEDICAL_QA = Suite(
    domain="medical-qa",
    version="1",
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


def load_items(bench: Benchmark, rows: list[dict] | None = None, seed: int = 0) -> list[MCQItem]:
    """Convert benchmark rows to items, sampling `bench.size` of them if there are more."""
    if rows is None:
        from datasets import load_dataset

        rows = load_dataset(bench.hf_id, bench.config, split=bench.split, revision=bench.revision)
    items = sorted((bench.to_item(row, i) for i, row in enumerate(rows)), key=lambda x: x.item_id)
    if len(items) > bench.size:
        items = sorted(random.Random(seed).sample(items, bench.size), key=lambda x: x.item_id)
    return items


def load_suite_items(suite: Suite) -> SuiteItems:
    return {b.name: load_items(b) for b in suite.benchmarks}
