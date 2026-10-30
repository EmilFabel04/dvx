from dataclasses import replace

import pytest

from dvx.evals.registry import MCQItem, UnknownDomain, get_suite, load_items


def _bench(name):
    return next(b for b in get_suite("medical-qa").benchmarks if b.name == name)


def test_medical_suite():
    s = get_suite("medical-qa")
    assert s.version == "1"
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


@pytest.mark.slow
def test_load_medqa_from_hub():
    items = load_items(_bench("medqa"))
    assert len(items) == 500
    assert all(len(i.options) == 4 for i in items)
