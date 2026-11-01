"""Prompt rendering and supervised fine-tuning encoding.

One plain-text protocol for every model; chat templates are never applied.
"""

from dataclasses import dataclass

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from dvx.data import Example
from dvx.evals.registry import LETTERS, MCQItem

IGNORE = -100
MIN_MAX_LEN = 8
PAD_MULTIPLE = 64


@dataclass(frozen=True)
class EncodedExample:
    input_ids: list[int]
    labels: list[int]
    truncated: bool


def render_prompt(prompt: str, system: str | None = None) -> str:
    """SFT prompt text: the system text (if any) and the prompt joined by a blank line,
    then the blank line that separates the prompt from the response."""
    return "\n\n".join([system, prompt] if system else [prompt]) + "\n\n"


def _adds_bos(tokenizer: PreTrainedTokenizerBase, text: str) -> bool:
    """True if the tokenizer itself prepends BOS when special tokens are enabled."""
    bos = tokenizer.bos_token_id
    if bos is None:
        return False
    with_special = tokenizer(text, add_special_tokens=True).input_ids
    plain = tokenizer(text, add_special_tokens=False).input_ids
    return bool(with_special) and with_special[0] == bos and not (plain and plain[0] == bos)


def encode_prompt_text(tokenizer: PreTrainedTokenizerBase, text: str) -> list[int]:
    """Token ids for rendered prompt text, exactly as SFT encodes a prompt: BOS first
    if the tokenizer would add one itself."""
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if _adds_bos(tokenizer, text):
        ids = [tokenizer.bos_token_id, *ids]
    return ids


def _mcq_block(item: MCQItem) -> str:
    options = "".join(f"{LETTERS[i]}. {opt}\n" for i, opt in enumerate(item.options))
    return f"Question: {item.question}\n{options}Answer:"


def render_mcq(item: MCQItem, shots: tuple[MCQItem, ...] | list[MCQItem] = ()) -> str:
    """Plain-text few-shot prompt, lm-eval style: each shot's block followed by its answer
    letter and a blank line, then the item's block ending in "Answer:"."""
    solved = "".join(f"{_mcq_block(s)} {LETTERS[s.answer]}\n\n" for s in shots)
    return solved + _mcq_block(item)


def padded_width(longest: int, max_len: int) -> int:
    """Width to pad a batch to: its longest row rounded up to a multiple of PAD_MULTIPLE,
    capped at the encoder's max_len (never below the longest row).

    Few distinct widths let a device's caching allocator reuse freed blocks across
    batches instead of holding one set per width.
    """
    rounded = -(-longest // PAD_MULTIPLE) * PAD_MULTIPLE
    return min(rounded, max(max_len, longest))


def encode_sft(tokenizer: PreTrainedTokenizerBase, ex: Example, max_len: int) -> EncodedExample:
    if max_len < MIN_MAX_LEN:
        raise ValueError(f"max_len must be at least {MIN_MAX_LEN}, got {max_len}")
    prompt = encode_prompt_text(tokenizer, render_prompt(ex.prompt, ex.system))
    response = [
        *tokenizer(ex.response, add_special_tokens=False).input_ids,
        tokenizer.eos_token_id,
    ]
    truncated = len(prompt) + len(response) > max_len
    if truncated:
        cap = max_len - max_len // 4
        if len(response) > cap:
            response = [*response[: cap - 1], tokenizer.eos_token_id]
        room = max_len - len(response)
        prompt = prompt[-room:] if room > 0 else []
    return EncodedExample(
        input_ids=prompt + response,
        labels=[IGNORE] * len(prompt) + response,
        truncated=truncated,
    )
