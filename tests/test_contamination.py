from dvx.contamination import find_contamination
from dvx.data import Example
from dvx.evals.registry import MCQItem

QUESTION = (
    "A 45-year-old man presents with fever, cough, and shortness of breath after returning "
    "from a trip abroad; which of the following is the most likely diagnosis"
)


def _ex(line: int, prompt: str, response: str = "") -> Example:
    return Example(prompt=prompt, response=response, system=None, line=line)


def _item(benchmark: str, question: str, options: tuple[str, ...] = ("a", "b")) -> MCQItem:
    return MCQItem(benchmark=benchmark, item_id="1", question=question, options=options, answer=0)


def test_flags_shared_13gram():
    item = _item("medqa", QUESTION, ("Influenza", "Pneumonia"))
    words = QUESTION.split()
    assert len(words) >= 20
    snippet = " ".join(words[3:16]).upper() + "!!"
    examples = [_ex(0, "Nothing relevant here at all."), _ex(1, f"Q: {snippet} Answer?")]
    res = find_contamination(examples, [item])
    assert res.flagged_lines == [1]
    assert res.n_flagged == 1
    assert res.by_benchmark == {"medqa": 1}
    assert res.ngram == 13


def test_short_item_substring():
    item = _item("arc", "What is the capital", ("of France",))
    ex = _ex(5, "Tell me: what is the capital of France? Paris.")
    res = find_contamination([ex], [item])
    assert res.flagged_lines == [5]
    assert res.by_benchmark == {"arc": 1}


def test_short_item_respects_word_boundaries():
    item = _item("arc", "the cat sat", ("on",))
    res = find_contamination([_ex(1, "scat sat on")], [item])
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
