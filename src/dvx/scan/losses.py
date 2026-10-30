"""Per-example losses and per-token log-probabilities."""

import numpy as np
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from dvx.data import Example
from dvx.prompting import IGNORE, encode_sft


def _pad(seqs: list[list[int]], value: int, device: torch.device) -> torch.Tensor:
    width = max(len(s) for s in seqs)
    return torch.tensor([s + [value] * (width - len(s)) for s in seqs], device=device)


@torch.no_grad()
def response_losses(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    examples: list[Example],
    max_len: int = 512,
    batch_size: int = 8,
) -> list[float]:
    """Mean NLL (nats) per response token for each example."""
    device = next(model.parameters()).device
    encoded = [encode_sft(tokenizer, ex, max_len) for ex in examples]
    out: list[float] = []
    for i in range(0, len(encoded), batch_size):
        chunk = encoded[i : i + batch_size]
        ids = _pad([e.input_ids for e in chunk], tokenizer.pad_token_id, device)
        labels = _pad([e.labels for e in chunk], IGNORE, device)
        mask = _pad([[1] * len(e.input_ids) for e in chunk], 0, device)
        logits = model(input_ids=ids, attention_mask=mask).logits.float()
        logp = F.log_softmax(logits[:, :-1], dim=-1)
        tgt = labels[:, 1:]
        valid = tgt != IGNORE
        picked = logp.gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        nll = -(picked * valid).sum(-1) / valid.sum(-1).clamp(min=1)
        out.extend(nll.tolist())
    return out


@torch.no_grad()
def token_logprobs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    max_len: int = 1024,
    batch_size: int = 8,
) -> list[np.ndarray]:
    """Log-probs of tokens 1..n-1 of each text (default special tokens, truncated)."""
    device = next(model.parameters()).device
    encoded = [tokenizer(t, truncation=True, max_length=max_len).input_ids for t in texts]
    out: list[np.ndarray] = []
    for i in range(0, len(encoded), batch_size):
        chunk = encoded[i : i + batch_size]
        ids = _pad(chunk, tokenizer.pad_token_id, device)
        mask = _pad([[1] * len(c) for c in chunk], 0, device)
        logits = model(input_ids=ids, attention_mask=mask).logits.float()
        logp = F.log_softmax(logits[:, :-1], dim=-1)
        picked = logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        for row, c in zip(picked, chunk, strict=True):
            out.append(row[: max(len(c) - 1, 0)].cpu().numpy())
    return out
