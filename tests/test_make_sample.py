import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "make_medmcqa_sample.py"
_spec = importlib.util.spec_from_file_location("make_medmcqa_sample", _PATH)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def _row(**kw):
    row = {
        "id": "x",
        "question": "Which is right?",
        "opa": "alpha",
        "opb": "beta",
        "opc": "gamma",
        "opd": "delta",
        "cop": 2,
        "exp": "because",
    }
    row.update(kw)
    return row


def test_row_to_example():
    ex = mod.medmcqa_row_to_example(_row())
    assert set(ex) == {"prompt", "response"}
    assert ex["prompt"] == (
        "Which is right?\n\nA. alpha\nB. beta\nC. gamma\nD. delta\n\n"
        "Answer with the letter of the correct option."
    )
    assert ex["response"].startswith("C. gamma")
    assert "Explanation: because" in ex["response"]


def test_row_without_explanation():
    for exp in (None, "", "   "):
        ex = mod.medmcqa_row_to_example(_row(exp=exp))
        assert ex["response"] == "C. gamma"
