"""LoRA fine-tuning with token and FLOP accounting."""

import math
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from dvx.config import TrainConfig
from dvx.data import Example
from dvx.models import count_params, release_cached_memory, split_decoder_head
from dvx.prompting import IGNORE, EncodedExample, encode_sft, padded_width

_LOSS_CHUNK = 32  # sequence positions per vocabulary projection in the training loss


@dataclass(frozen=True)
class TrainStats:
    """Plain-Python training record, safe to serialize as JSON."""

    n_examples: int
    n_truncated: int
    tokens_trained: int
    steps: int
    wall_seconds: float
    est_flops: int
    final_loss: float
    seed: int


def _collate(
    batch: list[EncodedExample], pad_id: int, device: torch.device, max_len: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = padded_width(max(len(e.input_ids) for e in batch), max_len)
    ids, labels, mask = [], [], []
    for e in batch:
        pad = width - len(e.input_ids)
        ids.append(e.input_ids + [pad_id] * pad)
        labels.append(e.labels + [IGNORE] * pad)
        mask.append([1] * len(e.input_ids) + [0] * pad)
    return (
        torch.tensor(ids, device=device),
        torch.tensor(labels, device=device),
        torch.tensor(mask, device=device),
    )


def _chunk_nll_sum(head: nn.Module, hidden: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits = head(hidden).float()
    return F.cross_entropy(
        logits.flatten(0, 1), targets.flatten(), ignore_index=IGNORE, reduction="sum"
    )


def _sft_loss(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Mean next-token NLL over the non-IGNORE labels: HF's causal-LM loss.

    For a split-head model (see split_decoder_head) the decoder runs once, then the head
    and float32 cross-entropy run _LOSS_CHUNK positions at a time, each chunk under
    activation checkpointing. So the [B, T, V] logits, their float32 copy, log-softmax
    and gradient never exist whole: each chunk's logits are rebuilt in backward. Other
    models use the model's own loss.
    """
    split = split_decoder_head(model)
    if split is None:
        return model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False
        ).loss
    decoder, head = split
    hidden = decoder(
        input_ids=input_ids, attention_mask=attention_mask, use_cache=False
    ).last_hidden_state
    targets = labels[:, 1:]
    total = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, targets.shape[1], _LOSS_CHUNK):
        end = min(start + _LOSS_CHUNK, targets.shape[1])
        total = total + checkpoint(
            _chunk_nll_sum, head, hidden[:, start:end], targets[:, start:end], use_reentrant=False
        )
    return total / (targets != IGNORE).sum()


def finetune_lora(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    examples: list[Example],
    cfg: TrainConfig,
    seed: int,
) -> tuple[PeftModel, TrainStats]:
    """Fine-tune `model` with LoRA on `examples`; return the adapter model in eval mode.

    `model` is wrapped in place by peft, so pass a freshly loaded model.
    Examples are shuffled once with `random.Random(seed)`, then iterated for
    `cfg.epochs` passes in micro-batches of `cfg.micro_batch_size`, with
    gradients accumulated over `cfg.grad_accum` micro-batches per optimizer step.
    Each micro-batch's loss is HF's causal-LM loss, computed in chunks by `_sft_loss`.
    The learning rate warms up linearly for ceil(warmup_ratio * steps) steps and
    then decays linearly to 0. With `cfg.gradient_checkpointing`, decoder activations
    are recomputed in the backward pass instead of stored, which trades about one extra
    forward pass for memory and leaves the result unchanged.

    `est_flops` is the 6 * N * D approximation, where N is the parameter count of
    the base model only (measured before the LoRA wrapper adds adapter weights)
    and D is `tokens_trained`, the number of non-pad input tokens processed over
    all epochs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_params = count_params(model)
    if cfg.gradient_checkpointing:
        # Non-reentrant: frozen inputs need no input-grad hook (peft adds one anyway).
        # torch.utils.checkpoint preserves RNG state on recompute by default.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    peft_model = get_peft_model(
        model,
        LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=list(cfg.lora_target_modules),
            task_type="CAUSAL_LM",
        ),
    )

    encoded = [encode_sft(tokenizer, ex, cfg.max_seq_len) for ex in examples]
    n_truncated = sum(e.truncated for e in encoded)
    order = list(range(len(encoded)))
    random.Random(seed).shuffle(order)
    stream = [encoded[i] for _ in range(cfg.epochs) for i in order]

    micro = [
        stream[i : i + cfg.micro_batch_size] for i in range(0, len(stream), cfg.micro_batch_size)
    ]
    groups = [micro[i : i + cfg.grad_accum] for i in range(0, len(micro), cfg.grad_accum)]
    steps = math.ceil(len(examples) * cfg.epochs / (cfg.micro_batch_size * cfg.grad_accum))
    assert steps == len(groups)
    warmup = math.ceil(cfg.warmup_ratio * steps)

    def lr_scale(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, (steps - step) / max(1, steps - warmup))

    params = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)

    device = next(peft_model.parameters()).device
    pad_id = tokenizer.pad_token_id
    tokens_trained = 0
    final_loss = float("nan")
    start = time.perf_counter()
    peft_model.train()
    for group in groups:
        optimizer.zero_grad()
        step_loss = 0.0
        for batch in group:
            ids, labels, mask = _collate(batch, pad_id, device, cfg.max_seq_len)
            loss = _sft_loss(peft_model, ids, mask, labels)
            (loss / len(group)).backward()
            step_loss += loss.item() / len(group)
            tokens_trained += int(mask.sum().item())
        optimizer.step()
        scheduler.step()
        final_loss = float(step_loss)
    wall = time.perf_counter() - start
    peft_model.eval()
    release_cached_memory(device)

    stats = TrainStats(
        n_examples=len(examples),
        n_truncated=int(n_truncated),
        tokens_trained=tokens_trained,
        steps=steps,
        wall_seconds=wall,
        est_flops=6 * n_params * tokens_trained,
        final_loss=final_loss,
        seed=seed,
    )
    return peft_model, stats
