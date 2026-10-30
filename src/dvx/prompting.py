"""Prompt rendering and supervised fine-tuning encoding."""

from dataclasses import dataclass

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from dvx.data import Example

IGNORE = -100
MIN_MAX_LEN = 8


@dataclass(frozen=True)
class EncodedExample:
    input_ids: list[int]
    labels: list[int]
    truncated: bool


def render_prompt(
    tokenizer: PreTrainedTokenizerBase, prompt: str, system: str | None = None
) -> str:
    if tokenizer.chat_template:
        msgs = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
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


def _prompt_ids(tokenizer: PreTrainedTokenizerBase, ex: Example) -> list[int]:
    text = render_prompt(tokenizer, ex.prompt, ex.system)
    ids = tokenizer(text, add_special_tokens=False).input_ids
    # Chat templates already emit their own special tokens; on the plain path
    # add BOS only if the tokenizer would add it itself.
    if not tokenizer.chat_template and _adds_bos(tokenizer, text):
        ids = [tokenizer.bos_token_id, *ids]
    return ids


def encode_sft(tokenizer: PreTrainedTokenizerBase, ex: Example, max_len: int) -> EncodedExample:
    if max_len < MIN_MAX_LEN:
        raise ValueError(f"max_len must be at least {MIN_MAX_LEN}, got {max_len}")
    prompt = _prompt_ids(tokenizer, ex)
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
