"""Model registry, loading and memory release."""

import gc
from dataclasses import dataclass
from datetime import date

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


@dataclass(frozen=True)
class ModelInfo:
    model_id: str
    revision: str | None
    family: str
    params_b: float | None
    reference_id: str | None
    unseen_after: date | None
    pretrain_tokens: float | None


def _qwen3(size: str, revision: str, params_b: float, reference: str | None) -> ModelInfo:
    return ModelInfo(
        model_id=f"Qwen/Qwen3-{size}",
        revision=revision,
        family="qwen3",
        params_b=params_b,
        reference_id=f"Qwen/Qwen3-{reference}" if reference else None,
        unseen_after=date(2025, 5, 1),
        pretrain_tokens=36e12,
    )


REGISTRY: dict[str, ModelInfo] = {
    info.model_id: info
    for info in (
        _qwen3("0.6B", "c1899de289a04d12100db370d81485cdf75e47ca", 0.6, "1.7B"),
        _qwen3("1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e", 1.7, "4B"),
        _qwen3("4B", "1cfa9a7208912126459214e8b04321603b3df60c", 4.0, "8B"),
        _qwen3("8B", "b968826d9c46dd6066d109eabc6255188de91218", 8.0, None),
    )
}


def model_info(model_id: str) -> ModelInfo:
    if model_id in REGISTRY:
        return REGISTRY[model_id]
    return ModelInfo(model_id, None, "unknown", None, None, None, None)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(
    model_id: str, revision: str | None, device: str, dtype: torch.dtype | None = None
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    if dtype is None:
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=dtype)
    model.to(device)
    model.eval()
    return model, tokenizer


# Model types whose logits are exactly lm_head(final hidden state); others may transform
# logits after the head (e.g. softcapping), so they take the full-logits path.
SPLIT_HEAD_MODEL_TYPES = frozenset({"llama", "mistral", "qwen2", "qwen3"})


def split_decoder_head(model: PreTrainedModel) -> tuple[nn.Module, nn.Module] | None:
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


def count_params(model: PreTrainedModel) -> int:
    return sum(p.numel() for p in model.parameters())


def release_cached_memory(device: torch.device | str) -> None:
    """Return the memory PyTorch's caching allocator holds unused on `device` (MPS or
    CUDA) to the system. Live tensors are untouched; a no-op on CPU."""
    kind = torch.device(device).type
    if kind == "mps":
        torch.mps.empty_cache()
    elif kind == "cuda":
        torch.cuda.empty_cache()


def release(model: PreTrainedModel) -> None:
    """Free the model's parameter memory. The model is unusable afterwards."""
    first = next(model.parameters(), None)
    device = first.device if first is not None else torch.device("cpu")
    model.to("meta")
    gc.collect()
    release_cached_memory(device)
