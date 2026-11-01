"""Per-example losses and per-token log-probabilities."""

import numpy as np
import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from dvx.data import Example
from dvx.models import release_cached_memory
from dvx.prompting import IGNORE, encode_sft, padded_width

_CHUNK = 32  # sequence positions projected to the vocabulary, and upcast, at a time
# Model types whose logits are exactly lm_head(final hidden state); others may transform
# logits after the head (e.g. softcapping), so they take the full-logits path.
SPLIT_HEAD_MODEL_TYPES = frozenset({"llama", "mistral", "qwen2", "qwen3"})


def _pad(seqs: list[list[int]], value: int, width: int, device: torch.device) -> torch.Tensor:
    return torch.tensor([s + [value] * (width - len(s)) for s in seqs], device=device)


def _length_sorted_batches(lengths: list[int], batch_size: int) -> list[list[int]]:
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    return [order[i : i + batch_size] for i in range(0, len(order), batch_size)]


def _decoder_and_head(model: PreTrainedModel) -> tuple[nn.Module, nn.Module] | None:
    """The model's decoder and output embedding, if its logits are the head applied to
    the decoder's final hidden states; None if the model can't be split that way or its
    type isn't in SPLIT_HEAD_MODEL_TYPES."""
    if getattr(getattr(model, "config", None), "model_type", None) not in SPLIT_HEAD_MODEL_TYPES:
        return None
    try:
        decoder, head = model.get_decoder(), model.get_output_embeddings()
    except (AttributeError, NotImplementedError):
        return None
    if decoder is None or head is None or any(m is head for m in decoder.modules()):
        return None
    return decoder, head


def _target_logprobs(
    model: PreTrainedModel, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Log-prob of each next token, shape [B, T-1], in float32.

    The decoder runs once, and its final hidden states are projected to the
    vocabulary _CHUNK positions at a time, each chunk reduced to gather - logsumexp
    in float32. Neither full-sequence logits nor a float32 log_softmax over the
    vocabulary is materialized. Any other model (not in SPLIT_HEAD_MODEL_TYPES, or not
    splittable into decoder and output head) falls back to full-sequence logits, still
    upcast one chunk at a time.
    """
    split = _decoder_and_head(model)
    if split is None:
        states = model(input_ids=input_ids, attention_mask=attention_mask).logits
        head: nn.Module = nn.Identity()
    else:
        decoder, head = split
        states = decoder(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).last_hidden_state
    targets = input_ids[:, 1:]
    parts = []
    for start in range(0, targets.shape[1], _CHUNK):
        end = min(start + _CHUNK, targets.shape[1])
        chunk = head(states[:, start:end]).float()
        picked = chunk.gather(-1, targets[:, start:end].unsqueeze(-1)).squeeze(-1)
        parts.append(picked - torch.logsumexp(chunk, dim=-1))
    return torch.cat(parts, dim=1)


@torch.no_grad()
def response_losses(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    examples: list[Example],
    max_len: int = 512,
    batch_size: int = 8,
) -> list[float]:
    """Mean NLL (nats) per response token for each example.

    The model must already be in eval mode; this function does not change it.
    """
    device = next(model.parameters()).device
    encoded = [encode_sft(tokenizer, ex, max_len) for ex in examples]
    out = [0.0] * len(encoded)
    for idx in _length_sorted_batches([len(e.input_ids) for e in encoded], batch_size):
        chunk = [encoded[i] for i in idx]
        width = padded_width(max(len(e.input_ids) for e in chunk), max_len)
        ids = _pad([e.input_ids for e in chunk], tokenizer.pad_token_id, width, device)
        labels = _pad([e.labels for e in chunk], IGNORE, width, device)
        mask = _pad([[1] * len(e.input_ids) for e in chunk], 0, width, device)
        picked = _target_logprobs(model, ids, mask)
        valid = labels[:, 1:] != IGNORE
        nll = -(picked * valid).sum(-1) / valid.sum(-1).clamp(min=1)
        for i, v in zip(idx, nll.tolist(), strict=True):
            out[i] = v
    release_cached_memory(device)
    return out


@torch.no_grad()
def token_logprobs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    max_len: int = 1024,
    batch_size: int = 8,
) -> list[np.ndarray]:
    """Log-probs of tokens 1..n-1 of each text (default special tokens, truncated).

    The model must already be in eval mode; this function does not change it.
    """
    device = next(model.parameters()).device
    encoded = [tokenizer(t, truncation=True, max_length=max_len).input_ids for t in texts]
    out: list[np.ndarray] = [np.zeros(0, dtype=np.float32)] * len(encoded)
    for idx in _length_sorted_batches([len(e) for e in encoded], batch_size):
        chunk = [encoded[i] for i in idx]
        width = padded_width(max(len(c) for c in chunk), max_len)
        ids = _pad(chunk, tokenizer.pad_token_id, width, device)
        mask = _pad([[1] * len(c) for c in chunk], 0, width, device)
        picked = _target_logprobs(model, ids, mask)
        for i, row, c in zip(idx, picked, chunk, strict=True):
            out[i] = row[: max(len(c) - 1, 0)].cpu().numpy()
    release_cached_memory(device)
    return out
