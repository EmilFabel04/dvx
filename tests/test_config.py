import ast
import re
import sys
import tomllib
from pathlib import Path

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
    assert s.novelty_min_reference_below_target == 0.5
    assert s.novelty_min_paired_discrimination == 0.6
    assert (s.min_k, s.exposure_min_auc, s.calibration_per_side) == (0.2, 0.6, 200)
    assert (s.minhash_threshold, s.shingle_words, s.num_perm) == (0.8, 5, 128)
    assert (s.score_max_len, s.batch_size) == (1024, 8)
    assert s.eval_max_len == 3072  # longest 5-shot prompt in medical-qa v2: 2,658 tokens


ROOT = Path(__file__).resolve().parent.parent


def _third_party_imports() -> set[str]:
    names = set()
    for path in (ROOT / "src" / "dvx").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
    return names - set(sys.stdlib_module_names) - {"dvx"}


def _declared(pyproject: str) -> set[str]:
    deps = tomllib.loads(pyproject)["project"]["dependencies"]
    return {re.split(r"[<>=!~ \[]", d, maxsplit=1)[0].replace("-", "_").lower() for d in deps}


def test_every_third_party_import_is_a_declared_dependency():
    declared = _declared((ROOT / "pyproject.toml").read_text())
    assert _third_party_imports() <= declared, _third_party_imports() - declared


def test_gradient_checkpointing_on_by_default():
    assert TrainConfig().gradient_checkpointing is True
