"""LoRA fine-tuning with token and FLOP accounting."""

import math
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from dvx.config import TrainConfig
from dvx.data import Example
from dvx.models import count_params
from dvx.prompting import IGNORE, EncodedExample, encode_sft


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
    batch: list[EncodedExample], pad_id: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(len(e.input_ids) for e in batch)
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
    The learning rate warms up linearly for ceil(warmup_ratio * steps) steps and
    then decays linearly to 0.

    `est_flops` is the 6 * N * D approximation, where N is the parameter count of
    the base model only (measured before the LoRA wrapper adds adapter weights)
    and D is `tokens_trained`, the number of non-pad input tokens processed over
    all epochs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_params = count_params(model)
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
            ids, labels, mask = _collate(batch, pad_id, device)
            loss = peft_model(input_ids=ids, attention_mask=mask, labels=labels).loss
            (loss / len(group)).backward()
            step_loss += loss.item() / len(group)
            tokens_trained += int(mask.sum().item())
        optimizer.step()
        scheduler.step()
        final_loss = float(step_loss)
    wall = time.perf_counter() - start
    peft_model.eval()

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
