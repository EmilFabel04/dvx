from dataclasses import replace

import pytest
from fixtures import benchmark_rows, medmcqa_sample

from dvx.contamination import find_contamination
from dvx.data import Example
from dvx.evals.registry import LETTERS, MCQItem, get_suite, load_items
from dvx.prompting import render_mcq

QUESTION = (
    "A 45-year-old man presents with fever, cough, and shortness of breath after returning "
    "from a trip abroad; which of the following is the most likely diagnosis"
)


def _ex(line: int, prompt: str, response: str = "") -> Example:
    return Example(prompt=prompt, response=response, system=None, line=line)


def _item(benchmark: str, question: str, options: tuple[str, ...] = ("a", "b")) -> MCQItem:
    return MCQItem(benchmark=benchmark, item_id="1", question=question, options=options, answer=0)


def test_flags_most_of_an_items_13grams():
    item = _item("medqa", QUESTION, ("Influenza", "Pneumonia"))
    words = QUESTION.split()  # 26 words; 28 after punctuation splits "45-year-old"
    snippet = " ".join(words[2:22]).upper() + "!!"  # 10 of the question's 16 13-grams
    examples = [_ex(0, "Nothing relevant here at all."), _ex(1, f"Q: {snippet} Answer?")]
    res = find_contamination(examples, [item])
    assert res.flagged_lines == [1]
    assert res.n_flagged == 1
    assert res.by_benchmark == {"medqa": 1}
    assert res.ngram == 13


def test_a_shared_stock_13gram_does_not_flag():
    # USMLE stems share stock phrasing: one 13-gram of 16 is not the item.
    item = _item("medqa", QUESTION, ("Influenza", "Pneumonia"))
    # "... which of the following is the most likely diagnosis"
    stock = " ".join(QUESTION.split()[-13:])
    ex = _ex(1, f"A child presents with a rash after a camping trip abroad; {stock}?", "Lyme")
    assert find_contamination([ex], [item]).n_flagged == 0


def test_short_item_substring():
    item = _item("arc", "What is the capital", ("of France",))
    ex = _ex(5, "Tell me: what is the capital of France? Paris.")
    res = find_contamination([ex], [item])
    assert res.flagged_lines == [5]
    assert res.by_benchmark == {"arc": 1}


def test_short_item_respects_word_boundaries():
    item = _item("arc", "the cat sat on the", ("mat",))
    res = find_contamination([_ex(1, "scat sat on the mat")], [item])
    assert res.n_flagged == 0


def test_clean():
    item = _item("medqa", QUESTION)
    res = find_contamination([_ex(1, "completely unrelated text about cooking pasta")], [item])
    assert res.n_flagged == 0
    assert res.flagged_lines == []
    assert res.by_benchmark == {}


def test_counts_once_per_benchmark():
    items = [_item("medqa", QUESTION), _item("medqa", QUESTION + " today"), _item("mmlu", QUESTION)]
    res = find_contamination([_ex(2, QUESTION), _ex(1, QUESTION)], items)
    assert res.flagged_lines == [1, 2]
    assert res.by_benchmark == {"medqa": 2, "mmlu": 2}


def test_labelled_options_do_not_hide_a_short_question():
    # A 5-word question is matched only through question + options (16 words), and the
    # option letters between the options used to break every 13-gram of such an example.
    item = _item(
        "arc",
        "Which substance is a compound?",
        ("salted butter from cows", "liquid nitrogen gas", "pure hydrogen gas", "ammonia"),
    )
    for labels in (
        ["A.", "B.", "C.", "D."],
        ["(A)", "(B)", "(C)", "(D)"],
        ["A)", "B)", "C)", "D)"],
        ["a.", "b.", "c.", "d."],
    ):
        lines = [f"{lab} {opt}" for lab, opt in zip(labels, item.options, strict=True)]
        text = "\n".join([f"Question: {item.question}", *lines, "Answer:"])
        assert find_contamination([_ex(1, text, " D")], [item]).n_flagged == 1, labels


def test_question_of_six_to_twelve_words_flags_with_one_of_its_options():
    item = _item(
        "medmcqa", "Progenitor hematopoietic stem cells originate in", ("Bone marrow", "Thymus")
    )
    quiz = "Quiz: progenitor HEMATOPOIETIC stem cells originate in which organ?"
    hit = _ex(1, quiz, "Bone marrow")
    no_option = _ex(2, "Progenitor hematopoietic stem cells originate in which organ?", "Liver")
    near = _ex(3, "Progenitor hematopoietic stem cells mature in the thymus.", "")
    assert find_contamination([hit, no_option, near], [item]).flagged_lines == [1]


def test_a_generic_short_question_alone_does_not_flag():
    # An MMLU macroeconomics item asks just "Which of the following statements is true?"
    item = _item("macro", "Which of the following statements is true?", ("M2 is smaller than M1.",))
    vignette = _ex(1, "A boy has jaundice. Which of the following statements is true?", "B. HbS")
    copied = _ex(2, "Which of the following statements is true?\nA. M2 is smaller than M1.", "A")
    assert find_contamination([vignette, copied], [item]).flagged_lines == [2]


def test_question_of_thirteen_words_or_more_flags_alone():
    item = _item("medqa", QUESTION, ("Influenza", "Pneumonia"))
    res = find_contamination([_ex(1, f"Case: {QUESTION}?", "Something else entirely")], [item])
    assert res.flagged_lines == [1]


def test_very_short_question_matches_only_with_its_options():
    item = _item("arc", "Which substance is a compound?", ("butter", "nitrogen", "ammonia"))
    alone = _ex(1, "Which substance is a compound? Think about it.", "")
    with_options = _ex(2, "Which substance is a compound?\nA. butter\nB. nitrogen\nC. ammonia", "C")
    assert find_contamination([alone, with_options], [item]).flagged_lines == [2]


def test_items_under_six_words_never_flag():
    item = _item("arc", "Hi", ("a", "b"))  # "hi a b": too short to tell from ordinary text
    assert find_contamination([_ex(1, "hi a b c", "Hi\nA. a\nB. b")], [item]).n_flagged == 0


def test_letter_stripping_keeps_ordinary_text_matching():
    question = "Night blindness is an early sign of deficiency of vitamin A."
    item = _item("medmcqa", question, ("True", "False"))
    ex = _ex(1, f"{question} True or false?", "True")
    assert find_contamination([ex], [item]).flagged_lines == [1]


# Real-world-shaped questions in every length class: 1-5 words (matched together with
# their options), 6-12 words (matched alone) and 13 or more (13-grams).
REGISTRY_SHAPED = (
    "Hypoglycemia?",
    "Which substance is a compound?",
    "Chlorine gas reacts most readily with",
    "What country is the rock group U2 from?",
    "Which of the following marker persists in chronic hepatitis and recurrence?",
    "IPC section dealing with illegal abortion with woman consent?",
    "Which of the following statements is true?",
    QUESTION,
)

UNRELATED = (  # MedMCQA-train-shaped (question, correct option), unrelated to the above
    ("Which of the following is the drug of choice for absence seizures?", "Ethosuximide"),
    ("All of the following are features of nephrotic syndrome except", "Hematuria"),
    ("Which nerve supplies the deltoid muscle?", "Axillary nerve"),
    ("The most common site of carcinoid tumour is", "Appendix"),
    ("A 30-year-old woman presents with fatigue and weight gain; which test first?", "TSH"),
    ("Chlorine is used in water treatment because it", "Kills bacteria"),
    ("Which of the following statements is true about digoxin?", "It inhibits Na/K ATPase"),
)


def _sample_script_format(line: int, question: str, options: tuple[str, ...]) -> Example:
    """As scripts/make_medmcqa_sample.py writes a row whose answer is option A."""
    labelled = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
    prompt = f"{question}\n\n{labelled}\n\nAnswer with the letter of the correct option."
    return _ex(line, prompt, f"A. {options[0]}")


def _inserted(items: list[MCQItem]) -> dict[str, list[Example]]:
    """Each item as a training example, in render_mcq format and in plain format."""
    return {
        "render_mcq": [
            _ex(i, render_mcq(it), f" {LETTERS[it.answer]}") for i, it in enumerate(items, 1)
        ],
        "plain": [
            _ex(i, it.question + "\n" + "\n".join(it.options), it.options[it.answer])
            for i, it in enumerate(items, 1)
        ],
        "sample_script": [
            replace(_sample_script_format(i, it.question, it.options), line=i)
            for i, it in enumerate(items, 1)
        ],
    }


@pytest.mark.parametrize("bench", get_suite("medical-qa").benchmarks, ids=lambda b: b.name)
def test_recall_on_registry_shaped_items(bench):
    questions = dict(enumerate(REGISTRY_SHAPED))
    n = len(REGISTRY_SHAPED)
    items = load_items(bench, rows=benchmark_rows(bench.name, bench.split, n, questions))
    for name, examples in _inserted(items).items():
        res = find_contamination(examples, items)
        missed = [it.question for i, it in enumerate(items, 1) if i not in res.flagged_lines]
        assert res.n_flagged / len(items) >= 0.95, (name, missed)
    unrelated = [
        _sample_script_format(i, q, (a, "Other", "Neither", "Both"))
        for i, (q, a) in enumerate(UNRELATED, 1)
    ]
    assert find_contamination(unrelated, items).n_flagged == 0


@pytest.mark.slow
def test_recall_on_every_registry_item_and_no_flags_on_the_medmcqa_sample():
    """Every registry item inserted verbatim is found; the MedMCQA train sample is clean."""
    from dvx.evals.registry import load_suite_items

    suite = get_suite("medical-qa")
    items = load_suite_items(suite)
    for b in suite.benchmarks:
        for name, examples in _inserted(items[b.name]).items():
            res = find_contamination(examples, items[b.name])
            assert res.n_flagged / len(examples) >= 0.95, (b.name, name, res.n_flagged)
    every_item = [it for b in suite.benchmarks for it in items[b.name]]
    res = find_contamination(medmcqa_sample(2000), every_item)
    assert res.n_flagged == 0, res
