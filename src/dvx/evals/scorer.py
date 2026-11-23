"""Letter-logit scoring of multiple-choice items."""

import numpy as np
import torch
from transformers import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from dvx.evals.registry import MCQItem
from dvx.prompting import encode_prompt_text, render_mcq

LETTERS = "ABCDEFGH"


def letter_token_ids(tokenizer: PreTrainedTokenizerBase, n: int) -> list[list[int]]:
    """Single-token ids for each of the first n letters, bare and space-prefixed."""
    out = []
    for letter in LETTERS[:n]:
        ids: list[int] = []
        for variant in (letter, f" {letter}"):
            enc = tokenizer(variant, add_special_tokens=False).input_ids
            if len(enc) == 1 and enc[0] not in ids:
                ids.append(enc[0])
        out.append(ids)
    return out


@torch.no_grad()
def score_mcq(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    items: list[MCQItem],
    batch_size: int = 8,
    max_len: int = 1024,
) -> np.ndarray:
    """Boolean correctness per item, in input order.

    The prediction is the letter with the highest next-token logit after the
    prompt. Prompts are left-truncated to max_len and left-padded, so the last
    position is the real last token of every row. The model must already be in
    eval mode; this function does not change it.
    """
    device = next(model.parameters()).device
    seqs = [encode_prompt_text(tokenizer, render_mcq(tokenizer, it))[-max_len:] for it in items]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    order = sorted(range(len(items)), key=lambda i: len(seqs[i]))
    correct = np.zeros(len(items), dtype=bool)
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        width = max(len(seqs[i]) for i in idx)
        ids = torch.tensor(
            [[pad_id] * (width - len(seqs[i])) + seqs[i] for i in idx], device=device
        )
        mask = torch.tensor(
            [[0] * (width - len(seqs[i])) + [1] * len(seqs[i]) for i in idx], device=device
        )
        last = model(input_ids=ids, attention_mask=mask, logits_to_keep=1).logits[:, -1].float()
        for row, i in enumerate(idx):
            letter_ids = letter_token_ids(tokenizer, len(items[i].options))
            scores = [last[row, lid].max().item() for lid in letter_ids]
            correct[i] = int(np.argmax(scores)) == items[i].answer
    return correct
