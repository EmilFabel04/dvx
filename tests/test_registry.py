from dataclasses import replace

import pytest
from fixtures import benchmark_rows

from dvx.evals.registry import (
    N_SHOTS,
    MCQItem,
    UnknownDomain,
    get_suite,
    load_items,
    load_shots,
    load_suite_items,
    load_suite_shots,
)


def _bench(name):
    return next(b for b in get_suite("medical-qa").benchmarks if b.name == name)


def test_medical_suite():
    s = get_suite("medical-qa")
    assert s.version == "2"
    assert [b.name for b in s.by_group("target")] == ["medqa", "medmcqa", "pubmedqa"]
    assert next(b.name for b in s.by_group("side_effect")) == "arc_challenge"
    assert len(s.by_group("adjacent")) == 4 and len(s.by_group("side_effect")) == 5


def test_unknown_domain():
    with pytest.raises(UnknownDomain, match="medical-qa"):
        get_suite("law")


@pytest.mark.parametrize(
    "name,row,expected",
    [
        (
            "medqa",
            {
                "question": "Q?",
                "answer": "two",
                "answer_idx": "B",
                "options": {"A": "one", "B": "two", "C": "three", "D": "four"},
            },
            MCQItem("medqa", "medqa-7", "Q?", ("one", "two", "three", "four"), 1),
        ),
        (
            "medmcqa",
            {
                "id": "abc",
                "question": "Q?",
                "opa": "a",
                "opb": "b",
                "opc": "c",
                "opd": "d",
                "cop": 3,
            },
            MCQItem("medmcqa", "abc", "Q?", ("a", "b", "c", "d"), 3),
        ),
        (
            "pubmedqa",
            {
                "pubid": 42,
                "question": "Does it?",
                "long_answer": "x",
                "context": {"contexts": ["First.", "Second."]},
                "final_decision": "maybe",
            },
            MCQItem(
                "pubmedqa",
                "pubmedqa-42",
                "Context: First. Second.\n\nQuestion: Does it?",
                ("yes", "no", "maybe"),
                2,
            ),
        ),
        (
            "mmlu_philosophy",
            {
                "question": "Q?",
                "subject": "philosophy",
                "choices": ["a", "b", "c", "d"],
                "answer": 2,
            },
            MCQItem("mmlu_philosophy", "mmlu-philosophy-7", "Q?", ("a", "b", "c", "d"), 2),
        ),
        (
            "arc_challenge",
            {
                "id": "Mercury_1",
                "question": "Q?",
                "choices": {"text": ["w", "x", "y", "z"], "label": ["1", "2", "3", "4"]},
                "answerKey": "3",
            },
            MCQItem("arc_challenge", "Mercury_1", "Q?", ("w", "x", "y", "z"), 2),
        ),
        (
            "arc_challenge",
            {
                "id": "Mercury_2",
                "question": "Q?",
                "choices": {"text": ["w", "x", "y", "z"], "label": ["A", "B", "C", "D"]},
                "answerKey": "D",
            },
            MCQItem("arc_challenge", "Mercury_2", "Q?", ("w", "x", "y", "z"), 3),
        ),
    ],
)
def test_converters(name, row, expected):
    assert _bench(name).to_item(row, 7) == expected


def _rows(n):
    return [
        {
            "id": f"id{i:02d}",
            "question": f"Q{i}",
            "opa": "a",
            "opb": "b",
            "opc": "c",
            "opd": "d",
            "cop": i % 4,
        }
        for i in range(n)
    ]


def test_load_items_sampling():
    small = replace(_bench("medmcqa"), size=3)
    rows = _rows(10)
    ids = {r["id"] for r in rows}
    items = load_items(small, rows=rows, seed=1)
    assert len(items) == 3
    assert [i.item_id for i in items] == sorted(i.item_id for i in items)
    assert {i.item_id for i in items} <= ids
    assert items == load_items(small, rows=rows, seed=1)
    others = {tuple(i.item_id for i in load_items(small, rows=rows, seed=s)) for s in range(2, 12)}
    assert others - {tuple(i.item_id for i in items)}
    assert len(load_items(small, rows=_rows(3), seed=1)) == 3
    assert len(load_items(small, rows=_rows(2), seed=1)) == 2


_MMLU_REV = "c30699e8356da336a370243923dbaf21066bb9fe"
_PINS = {
    "medqa": (
        "target",
        "GBaker/MedQA-USMLE-4-options",
        None,
        "test",
        "0fb93dd23a7339b6dcd27e241cb9b5eca62d4d18",
        500,
    ),
    "medmcqa": (
        "target",
        "openlifescienceai/medmcqa",
        None,
        "validation",
        "91c6572c454088bf71b679ad90aa8dffcd0d5868",
        500,
    ),
    "pubmedqa": (
        "target",
        "qiaojin/PubMedQA",
        "pqa_labeled",
        "train",
        "9001f2853fb87cab8d220904e0de81ac6973b318",
        500,
    ),
    "arc_challenge": (
        "side_effect",
        "allenai/ai2_arc",
        "ARC-Challenge",
        "test",
        "210d026faf9955653af8916fad021475a3f00453",
        500,
    ),
    **{
        f"mmlu_{s}": (g, "cais/mmlu", s, "test", _MMLU_REV, 200)
        for g, subjects in {
            "adjacent": [
                "college_biology",
                "high_school_biology",
                "college_chemistry",
                "high_school_chemistry",
            ],
            "side_effect": [
                "high_school_mathematics",
                "philosophy",
                "high_school_macroeconomics",
                "miscellaneous",
            ],
        }.items()
        for s in subjects
    },
}


def test_pinned_benchmark_values():
    benchmarks = get_suite("medical-qa").benchmarks
    assert {b.name for b in benchmarks} == set(_PINS)
    for b in benchmarks:
        assert (b.group, b.hf_id, b.config, b.split, b.revision, b.size) == _PINS[b.name], b.name


# Few-shot rows come from another split of the same pinned dataset where one exists;
# PubMedQA pqa_labeled has only "train", so its shots are rows not sampled as eval items.
_SHOT_SPLITS = {
    "medqa": "train",
    "medmcqa": "train",
    "pubmedqa": None,
    "arc_challenge": "train",
    **{name: "dev" for name in _PINS if name.startswith("mmlu_")},
}


def test_shot_splits():
    assert N_SHOTS == 5
    assert {b.name: b.shot_split for b in get_suite("medical-qa").benchmarks} == _SHOT_SPLITS


def _eval_and_shot_rows(bench, shot_overrides=None):
    """Hub-format rows for one benchmark: (eval split rows, rows its shots are drawn from)."""
    eval_rows = benchmark_rows(bench.name, bench.split, 30)
    if bench.shot_split is None:
        return eval_rows, eval_rows
    return eval_rows, benchmark_rows(bench.name, bench.shot_split, 12, shot_overrides)


@pytest.mark.parametrize("bench", get_suite("medical-qa").benchmarks, ids=lambda b: b.name)
def test_shots_are_disjoint_from_eval_items(bench):
    eval_rows, shot_rows = _eval_and_shot_rows(bench)
    items = load_items(replace(bench, size=10), rows=eval_rows)
    shots = load_shots(bench, items, rows=shot_rows)
    assert len(shots) == N_SHOTS
    assert all(s.benchmark == bench.name for s in shots)
    assert {s.question for s in shots}.isdisjoint(it.question for it in items)
    if bench.shot_split is None:  # same split: the shots must be rows left out of the sample
        assert {s.item_id for s in shots}.isdisjoint(it.item_id for it in items)


@pytest.mark.parametrize("bench", get_suite("medical-qa").benchmarks, ids=lambda b: b.name)
def test_shots_are_deterministic_and_seeded(bench):
    eval_rows, shot_rows = _eval_and_shot_rows(bench)
    items = load_items(replace(bench, size=10), rows=eval_rows)
    first = load_shots(bench, items, rows=shot_rows)
    assert first == load_shots(bench, items, rows=shot_rows)
    assert first == load_shots(bench, items, rows=shot_rows, seed=0)
    others = {tuple(load_shots(bench, items, rows=shot_rows, seed=s)) for s in range(1, 6)}
    assert others - {tuple(first)}


def test_shots_skip_rows_that_repeat_an_eval_question():
    bench = _bench("medmcqa")
    eval_rows = benchmark_rows("medmcqa", "validation", 30)
    items = load_items(replace(bench, size=10), rows=eval_rows)
    repeats = {i: items[i % len(items)].question.upper() + "  " for i in range(7)}
    shot_rows = benchmark_rows("medmcqa", "train", 12, repeats)  # rows 0-6 restate eval items
    shots = load_shots(bench, items, rows=shot_rows)
    assert sorted(s.item_id for s in shots) == [f"train-{i:04d}" for i in range(7, 12)]


def test_too_few_shot_rows_raises():
    bench = _bench("arc_challenge")
    items = load_items(bench, rows=benchmark_rows("arc_challenge", "test", 3))
    with pytest.raises(ValueError, match="5 shots"):
        load_shots(bench, items, rows=benchmark_rows("arc_challenge", "train", 4))


@pytest.mark.slow
def test_hub_shots_are_disjoint_and_pinned_by_seed():
    suite = get_suite("medical-qa")
    items = load_suite_items(suite)
    shots = load_suite_shots(suite, items)
    for b in suite.benchmarks:
        assert len(shots[b.name]) == N_SHOTS, b.name
        assert {s.question for s in shots[b.name]}.isdisjoint(i.question for i in items[b.name])
        if b.shot_split is None:
            ids = {s.item_id for s in shots[b.name]}
            assert ids.isdisjoint(i.item_id for i in items[b.name]), b.name
    assert shots == load_suite_shots(suite, items)


@pytest.mark.slow
def test_load_medqa_from_hub():
    items = load_items(_bench("medqa"))
    assert len(items) == 500
    assert all(len(i.options) == 4 for i in items)
