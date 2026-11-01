"""Frozen configuration defaults for dvx."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TrainConfig:
    lr: float = 2e-4
    epochs: int = 1
    micro_batch_size: int = 4
    grad_accum: int = 2
    max_seq_len: int = 512
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    gradient_checkpointing: bool = True


@dataclass(frozen=True)
class ScanConfig:
    known_margin: float = 0.1
    noise_percentile: float = 10.0
    mismatched_n: int = 200
    # Novelty reliability gate: below either, novelty is "inconclusive".
    novelty_min_reference_below_target: float = 0.5
    novelty_min_paired_discrimination: float = 0.6
    min_k: float = 0.2
    exposure_min_auc: float = 0.6
    minhash_threshold: float = 0.8
    shingle_words: int = 5
    num_perm: int = 128
    calibration_per_side: int = 200
    score_max_len: int = 1024
    batch_size: int = 8
    # Few-shot eval prompts: the longest in medical-qa v2 is 2,658 tokens (PubMedQA;
    # MedQA 1,979). Longer prompts are left-truncated and counted.
    eval_max_len: int = 3072


@dataclass(frozen=True)
class Profile:
    name: str
    train_cap: int
    seeds: tuple[int, ...]
    measure_cap: int
    baseline_margin: float
    bootstrap_samples: int
    train: TrainConfig = field(default_factory=TrainConfig)
    scan: ScanConfig = field(default_factory=ScanConfig)


PROFILES: dict[str, Profile] = {
    "quick": Profile(
        name="quick",
        train_cap=2000,
        seeds=(0,),
        measure_cap=20_000,
        baseline_margin=0.05,
        bootstrap_samples=1000,
    ),
}
