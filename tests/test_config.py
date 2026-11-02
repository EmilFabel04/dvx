from dvx import __version__
from dvx.config import PROFILES, ScanConfig, TrainConfig


def test_version():
    assert __version__ == "0.1.0"


def test_quick_profile():
    p = PROFILES["quick"]
    assert (p.name, p.train_cap, p.seeds, p.measure_cap) == ("quick", 2000, (0,), 20_000)
    assert (p.baseline_margin, p.bootstrap_samples) == (0.05, 1000)


def test_train_defaults():
    t = TrainConfig()
    assert (t.lr, t.epochs, t.micro_batch_size, t.grad_accum) == (2e-4, 1, 4, 2)
    assert (t.max_seq_len, t.warmup_ratio, t.weight_decay) == (512, 0.03, 0.0)
    assert (t.lora_r, t.lora_alpha, t.lora_dropout) == (16, 32, 0.05)
    assert t.lora_target_modules == (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def test_scan_defaults():
    s = ScanConfig()
    assert (s.known_margin, s.noise_percentile, s.mismatched_n) == (0.1, 10.0, 200)
    assert (s.min_k, s.exposure_min_auc, s.calibration_per_side) == (0.2, 0.6, 200)
    assert (s.minhash_threshold, s.shingle_words, s.num_perm) == (0.8, 5, 128)
    assert (s.score_max_len, s.batch_size) == (1024, 8)
