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
    b = _bench("medmcqa")
    small = type(b)(**{**b.__dict__, "size": 3})
    items = load_items(small, rows=_rows(10), seed=1)
    assert len(items) == 3
    assert [i.item_id for i in items] == sorted(i.item_id for i in items)
    assert items == load_items(small, rows=_rows(10), seed=1)
    assert len(load_items(small, rows=_rows(2), seed=1)) == 2


@pytest.mark.slow
def test_load_medqa_from_hub():
    items = load_items(_bench("medqa"))
    assert len(items) == 500
    assert all(len(i.options) == 4 for i in items)
