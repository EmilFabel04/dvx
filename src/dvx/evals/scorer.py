"""Few-shot letter-logit scoring of multiple-choice items."""

from dataclasses import dataclass

import numpy as np
import torch
from transformers import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from dvx.evals.registry import LETTERS, N_SHOTS, MCQItem
from dvx.models import release_cached_memory
from dvx.prompting import encode_prompt_text, render_mcq

PROTOCOL = f"plain-text {N_SHOTS}-shot log-likelihood letter scoring"


@dataclass(frozen=True)
class MCQScores:
    """Per item, in input order: the predicted option index and whether it is the answer.
    `n_truncated` counts prompts left-truncated to the scorer's max_len."""

    predicted: np.ndarray
    correct: np.ndarray
    n_truncated: int


def letter_token_ids(tokenizer: PreTrainedTokenizerBase, n: int) -> list[int]:
    """Token id of " A", " B", ... for the first n letters, as each follows "Answer:".

    Raises ValueError if a letter is not exactly one token there.
    """
    prefix = tokenizer("Answer:", add_special_tokens=False).input_ids
    out = []
    for letter in LETTERS[:n]:
        ids = tokenizer(f"Answer: {letter}", add_special_tokens=False).input_ids
        if len(ids) != len(prefix) + 1 or ids[:-1] != prefix:
            raise ValueError(
                f"the tokenizer does not encode ' {letter}' after 'Answer:' as one token"
            )
        out.append(ids[-1])
    return out


@torch.no_grad()
def score_mcq(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    items: list[MCQItem],
    shots: tuple[MCQItem, ...] | list[MCQItem] = (),
    batch_size: int = 8,
    max_len: int = 4096,
) -> MCQScores:
    """Score each item's few-shot prompt (see render_mcq) by its next-token logits.

    The prediction is the option whose " A"/" B"/... token has the highest logit after
    the final "Answer:". Prompts longer than max_len are left-truncated (their earliest
    shots are cut) and counted. Rows are left-padded, so the last position is the real
    last token of every row. The model must already be in eval mode; this function does
    not change it.
    """
    device = next(model.parameters()).device
    full = [encode_prompt_text(tokenizer, render_mcq(it, shots)) for it in items]
    n_truncated = sum(len(s) > max_len for s in full)
    seqs = [s[-max_len:] for s in full]
    letters = {k: letter_token_ids(tokenizer, k) for k in {len(it.options) for it in items}}
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    order = sorted(range(len(items)), key=lambda i: len(seqs[i]))
    predicted = np.zeros(len(items), dtype=np.int64)
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        width = max(len(seqs[i]) for i in idx)
        ids = torch.tensor(
            [[pad_id] * (width - len(seqs[i])) + seqs[i] for i in idx], device=device
        )
        mask = torch.tensor(
            [[0] * (width - len(seqs[i])) + [1] * len(seqs[i]) for i in idx], device=device
        )
        last = model(
            input_ids=ids, attention_mask=mask, logits_to_keep=1, use_cache=False
        ).logits[:, -1].float()
        for row, i in enumerate(idx):
            option_ids = torch.tensor(letters[len(items[i].options)], device=device)
            predicted[i] = int(torch.argmax(last[row, option_ids]).item())
    release_cached_memory(device)
    answers = np.array([it.answer for it in items], dtype=np.int64)
    return MCQScores(predicted=predicted, correct=predicted == answers, n_truncated=n_truncated)
