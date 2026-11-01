"""Prompt rendering and supervised fine-tuning encoding."""

from dataclasses import dataclass

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from dvx.data import Example
from dvx.evals.registry import MCQItem

IGNORE = -100
MIN_MAX_LEN = 8
PAD_MULTIPLE = 64


@dataclass(frozen=True)
class EncodedExample:
    input_ids: list[int]
    labels: list[int]
    truncated: bool


def _apply_chat(tokenizer: PreTrainedTokenizerBase, msgs: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def render_prompt(
    tokenizer: PreTrainedTokenizerBase, prompt: str, system: str | None = None
) -> str:
    if tokenizer.chat_template:
        msgs = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        return _apply_chat(tokenizer, msgs)
    prefix = f"{system}\n\n" if system else ""
    return f"{prefix}### Question:\n{prompt}\n\n### Answer:\n"


def _adds_bos(tokenizer: PreTrainedTokenizerBase, text: str) -> bool:
    """True if the tokenizer itself prepends BOS when special tokens are enabled."""
    bos = tokenizer.bos_token_id
    if bos is None:
        return False
    with_special = tokenizer(text, add_special_tokens=True).input_ids
    plain = tokenizer(text, add_special_tokens=False).input_ids
    return bool(with_special) and with_special[0] == bos and not (plain and plain[0] == bos)


def encode_prompt_text(tokenizer: PreTrainedTokenizerBase, text: str) -> list[int]:
    """Token ids for rendered prompt text, exactly as SFT encodes a prompt."""
    ids = tokenizer(text, add_special_tokens=False).input_ids
    # Chat templates already emit their own special tokens; on the plain path
    # add BOS only if the tokenizer would add it itself.
    if not tokenizer.chat_template and _adds_bos(tokenizer, text):
        ids = [tokenizer.bos_token_id, *ids]
    return ids


def render_mcq(tokenizer: PreTrainedTokenizerBase, item: MCQItem) -> str:
    options = "\n".join(f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(item.options))
    content = f"{item.question}\n\n{options}\n\nAnswer with the letter of the correct option."
    if tokenizer.chat_template:
        return _apply_chat(tokenizer, [{"role": "user", "content": content}])
    return f"{content}\nAnswer:"


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
    prompt = encode_prompt_text(tokenizer, render_prompt(tokenizer, ex.prompt, ex.system))
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
