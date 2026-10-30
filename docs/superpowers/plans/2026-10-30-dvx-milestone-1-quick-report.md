# dvx Milestone 1 (Quick Report) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `dvx run DATA --model M --domain medical-qa --profile quick` produces a reproducible Data Impact Report and a corpus record. The report covers part 1 (novelty scan, exposure, duplicates), contamination detection, and part 2 (impact test with one seed, tuned and control, with bootstrap CIs). `dvx scan` runs the no-training subset.

**Architecture:** A `src/`-layout Python package. Each report part is a small module with pure-data result dataclasses. A step-cached pipeline (`run/pipeline.py`) orchestrates the modules: each step writes JSON atomically into `.dvx/runs/<run_id>/steps/`, so `dvx resume` can continue an interrupted run. Models are loaded one at a time and released between steps, to fit in 16 GB.

**Tech Stack:**
- Python 3.12 and uv
- PyTorch (MPS / CUDA / CPU)
- Hugging Face `transformers` (≥4.51, for Qwen3), `peft`, `datasets`
- numpy and `datasketch`
- pytest and ruff

**Spec:** `docs/superpowers/specs/2026-10-30-data-impact-engine-design.md`

**Scope:** this is milestone 1 of 3. Each milestone gets its own plan, written once the previous one is done.
- **M1 (this plan):** everything the `quick` profile needs, plus real M5 timings.
- **M2:** the `full` profile.
  - 3 seeds.
  - Scaling curve with matched control sizes.
  - Clean-gain rerun.
  - Cluster contributions.
  - Conversion-anchor ingredients (sibling-model evals).

  M2's run plan depends on the timings M1 measures.
- **M3:** validation.
  - The known-answer tests from spec §8.
  - The ~8B transfer check on a rented GPU.

## Global Constraints

- Python `>=3.12`. One codebase runs on `mps` (Apple M5, 16 GB), `cuda` and `cpu`.
- Input is text only: JSONL with `{"prompt","response"}` or single-turn `{"messages":[...]}`.
- No price or currency output anywhere.
- Every reported number appears with its model and context. Deltas are in **percentage points with a 95% confidence interval**.
- Reproducibility: record the dataset content hash, model ID and pinned revision, seeds, engine version and library versions.
- Datasets are measured in full up to **20,000 examples**. Above that, a fixed random sample is used, and the report says "measured on a sample of N of M examples".
- Evaluation suites come from a per-domain registry with pinned versions. v1 registers only `medical-qa`.
- The baseline check stops the run if the base model isn't clearly above chance on the target suite.
- The exposure estimate always carries its calibration AUC. It is "inconclusive" when the AUC is below the threshold or calibration is unavailable.
- Models: the Qwen3 family (Apache-2.0) at 0.6B, 1.7B, 4B and 8B. Tests use `hf-internal-testing/tiny-random-LlamaForCausalLM` on CPU.
- Tests marked `@pytest.mark.slow` (real models, Hub downloads of benchmarks) are excluded by default.

## Review Focus

1. **Malformed JSONL.** This covers invalid JSON, missing keys, non-string fields, multi-turn chats, empty responses and empty files. Each gives a `DatasetFormatError` naming the line number, and the CLI exits with code 2 and no traceback. Tests are in Task 2 and Task 15.
2. **Examples longer than `max_seq_len` (512).** They are truncated rather than crashing or silently vanishing. The prompt is trimmed from the left, the response from the right, and the response always keeps tokens. Truncated examples are counted and reported. Tests are in Task 4 and Task 11.
3. **A model that isn't in the registry,** or that has no reference model or `unseen_after` date. The run still completes: novelty is "unavailable" and exposure "inconclusive", each with a reason. Test in Task 14.
4. **A run interrupted mid-step** (MPS out-of-memory, Ctrl-C). `dvx resume` reuses the completed steps and never trusts a partially written step file. Tests are in Task 14.
5. **No network, or Wikipedia failing during exposure calibration.** Exposure becomes "inconclusive (calibration unavailable)" and the run continues. Tests are in Task 6 and Task 14.

---

### Task 1: Project scaffold and configuration

**Files:**
- Create:
  - `pyproject.toml`
  - `.python-version` (contents `3.12`)
  - `src/dvx/__init__.py`
  - `src/dvx/config.py`
  - `tests/test_config.py`
  - `tests/conftest.py` (empty for now)
  - `.github/workflows/ci.yml`
- Modify:
  - `.gitignore`: remove the `.python-version` line and add `data/`.
  - `README.md`: add a "Development" section with `uv sync --extra dev`, `uv run pytest` and `uv run ruff check .`.

**Interfaces:**
- Produces from `dvx/__init__.py`: `__version__ = "0.1.0"`.
- Produces from `dvx/config.py`, all frozen dataclasses with defaults:
  - `TrainConfig(lr, epochs, micro_batch_size, grad_accum, max_seq_len, warmup_ratio, weight_decay, lora_r, lora_alpha, lora_dropout, lora_target_modules: tuple[str, ...])`
  - `ScanConfig(known_margin, noise_percentile, mismatched_n, min_k, exposure_min_auc, minhash_threshold, shingle_words, num_perm, calibration_per_side, score_max_len, batch_size)`
  - `Profile(name, train_cap, seeds: tuple[int, ...], measure_cap, baseline_margin, bootstrap_samples, train: TrainConfig, scan: ScanConfig)`
  - `PROFILES: dict[str, Profile]`, containing only `"quick"` in M1.

`pyproject.toml`:
- Build backend: hatchling.
- `requires-python = ">=3.12"`.
- Dependencies: `torch>=2.6`, `transformers>=4.51`, `peft>=0.15`, `datasets>=3.0`, `numpy>=1.26`, `datasketch>=1.6`.
- Extra `dev = ["pytest>=8", "ruff>=0.6"]`.
- Script `dvx = "dvx.cli:main"`.
- pytest: `addopts = "-m 'not slow'"`, `markers = ["slow: real models or Hub downloads"]`, `testpaths = ["tests"]`.
- Ruff: `line-length = 100`.

CI runs on `ubuntu-latest` with `astral-sh/setup-uv`:
1. `uv venv --python 3.12`.
2. `uv pip install torch --index-url https://download.pytorch.org/whl/cpu`. This keeps the 2 GB CUDA build out of CI. Local and cloud setups use `uv sync`.
3. `uv pip install -e ".[dev]"`.
4. `.venv/bin/ruff check .`.
5. `.venv/bin/pytest`.
6. Cache `~/.cache/huggingface`.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2:** Run `uv sync --extra dev && uv run pytest tests/test_config.py -v`. Expected: FAIL with an import error.
- [ ] **Step 3:** Create `pyproject.toml`, the package and `config.py` with the values above.
- [ ] **Step 4:** Run `uv run pytest -v && uv run ruff check .`. Expected: 4 passed and ruff clean.
- [ ] **Step 5:** Commit, including `uv.lock`: `git commit -m "feat: project scaffold, config defaults, CI"`.

---

### Task 2: Dataset loading, sampling, fingerprint, control data

**Files:**
- Create: `src/dvx/data.py`, `tests/test_data.py`

**Interfaces:**
- Produces:
  - `Example(prompt: str, response: str, system: str | None, line: int)`, frozen.
  - `class DatasetFormatError(ValueError)`.
  - `load_examples(path: Path) -> list[Example]`.
  - `fingerprint(examples: list[Example]) -> str`: a 64-character SHA-256 hex digest over canonical JSON lines of `(system, prompt, response)`, in order.
  - `sample_examples(examples, cap: int, seed: int) -> list[Example]`: everything if `len <= cap`. Otherwise `random.Random(seed).sample` of `cap`, returned sorted by `line`.
  - `load_control_examples(n: int, seed: int, rows: Iterable[dict] | None = None) -> list[Example]`. Builds Dolly examples: `prompt = instruction` plus `"\n\n" + context` when the context is non-empty, and `response = response`. Takes a seeded sample of `n`, and raises `ValueError` if `n` exceeds the available rows.
  - `DOLLY_ID = "databricks/databricks-dolly-15k"`, `DOLLY_REVISION = "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"`. When `rows` is None, load the `train` split at this revision.

Parsing rules:
- Blank lines are skipped. `line` is 1-based and refers to the physical line.
- In `messages` format, an optional leading `system`, then exactly one `user` turn and one `assistant` turn.
- Error messages must contain `f"line {n}"` plus one of:
  - `"invalid JSON"`
  - `"missing 'prompt'/'response' or 'messages'"`
  - `"multi-turn conversations are not supported in v1"`
  - `"must be strings"`
  - `"empty response"`
- A file with no examples raises `"no examples"`.

- [ ] **Step 1: Write the failing tests**

```python
def test_prompt_response(tmp_path):  # two lines -> two Examples with line 1, 2
def test_messages_with_system(tmp_path):
    # {"messages":[{"role":"system","content":"s"},{"role":"user","content":"u"},
    #              {"role":"assistant","content":"a"}]}
    assert load_examples(p) == [Example(prompt="u", response="a", system="s", line=1)]
def test_blank_lines_skipped(tmp_path):  # "\n" + valid line -> one Example with line == 2
@pytest.mark.parametrize("bad, msg", [
    ("{not json", "invalid JSON"),
    ('{"prompt": "p"}', "missing"),
    (MULTI_TURN_LINE, "multi-turn"),
    ('{"prompt": 1, "response": "r"}', "must be strings"),
    ('{"prompt": "p", "response": "  "}', "empty response")])
def test_errors_name_line(tmp_path, bad, msg):  # valid line 1 + bad line 2
    with pytest.raises(DatasetFormatError) as e: load_examples(p)
    assert "line 2" in str(e.value) and msg in str(e.value)
def test_empty_file(tmp_path):  # raises DatasetFormatError matching "no examples"
def test_fingerprint():  # 64 hex chars; equal for equal lists; differs when order is reversed
def test_sample_under_cap():  # sample_examples(exs, 10, 0) == exs
def test_sample_over_cap():  # 100 -> 10; same seed equal; lines ascending; seed 1 differs
def test_control_from_rows():
    rows = [{"instruction": "i", "context": "", "response": "r", "category": "x"},
            {"instruction": "i2", "context": "c2", "response": "r2", "category": "y"}]
    assert {e.prompt for e in load_control_examples(2, 0, rows)} == {"i", "i2\n\nc2"}
    with pytest.raises(ValueError): load_control_examples(3, 0, rows)
```

- [ ] **Step 2:** Run `uv run pytest tests/test_data.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement `data.py`.
- [ ] **Step 4:** Run `uv run pytest tests/test_data.py -v`. Expected: all pass.
- [ ] **Step 5:** Commit: `feat: dataset loading, sampling, fingerprint, control data`.

---

### Task 3: Duplicate detection

**Files:**
- Create: `src/dvx/scan/__init__.py`, `src/dvx/scan/duplicates.py`, `tests/test_duplicates.py`

**Interfaces:**
- Consumes: `Example`.
- Produces:
  - `DuplicateStats(n, exact_dup, near_dup, exact_fraction, near_fraction)`, frozen.
  - `normalize_text(s: str) -> str`: Unicode NFKC, lowercase, whitespace collapsed to a single space, stripped.
  - `duplicate_stats(examples, threshold=0.8, shingle_words=5, num_perm=128) -> DuplicateStats`.

How it works:
- The text compared is `prompt + "\n" + response`.
- `exact_dup` counts examples whose normalized text equals an earlier one.
- `near_dup` counts examples that aren't exact duplicates but whose `datasketch.MinHashLSH` query (word 5-gram shingles, MinHash `seed=1`) hits an earlier example at the threshold.

- [ ] **Step 1: Write the failing tests**

```python
def test_exact_after_normalization():  # ("Hello  World","x") and ("hello world","X") -> exact_dup == 1
def test_near_duplicate():  # a 60-word text and a copy with one word changed -> near_dup == 1, exact_dup == 0
def test_distinct():  # two unrelated 60-word texts -> (0, 0)
def test_fractions():  # 4 examples, one exact copy -> exact_fraction == 0.25
```

- [ ] **Step 2:** Run `uv run pytest tests/test_duplicates.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the same command. Expected: PASS.
- [ ] **Step 5:** Commit: `feat: exact and near-duplicate detection`.

---

### Task 4: Model registry, loading, SFT encoding

**Files:**
- Create: `src/dvx/models.py`, `src/dvx/prompting.py`, `tests/test_models.py`, `tests/test_prompting.py`
- Modify: `tests/conftest.py`. Add `TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"`, `TINY_REV = "9fb191250dd56d0ba7ec9785a025ed29c03d5998"`, and a function-scoped fixture `tiny` returning `load_model(TINY, TINY_REV, "cpu")`.

**Interfaces:**
- Produces from `models.py`:
  - `ModelInfo(model_id, revision: str | None, family: str, params_b: float | None, reference_id: str | None, unseen_after: date | None, pretrain_tokens: float | None)`, frozen.
  - `REGISTRY: dict[str, ModelInfo]`.
  - `model_info(model_id: str) -> ModelInfo`. An unknown ID gives `family="unknown"` and every other field None.
  - `pick_device() -> str`: `"cuda"`, then `"mps"`, then `"cpu"`.
  - `load_model(model_id, revision, device, dtype=None) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]`. The default dtype is bfloat16 on cuda/mps and float32 on cpu. If the tokenizer has no pad token, use the EOS token. The model is put in eval mode.
  - `count_params(model) -> int`.
  - `release(model) -> None`: delete, `gc.collect()`, then `torch.mps.empty_cache()` or `torch.cuda.empty_cache()` depending on device.
- Produces from `prompting.py`:
  - `EncodedExample(input_ids: list[int], labels: list[int], truncated: bool)`.
  - `render_prompt(tokenizer, prompt: str, system: str | None = None) -> str`.
  - `encode_sft(tokenizer, ex: Example, max_len: int) -> EncodedExample`.

Registry entries. All Qwen3 entries have `family="qwen3"`, `unseen_after=date(2025, 5, 1)` and `pretrain_tokens=36e12`.

| Model | Revision | `params_b` | `reference_id` |
|---|---|---|---|
| `Qwen/Qwen3-0.6B` | `c1899de289a04d12100db370d81485cdf75e47ca` | 0.6 | `Qwen/Qwen3-1.7B` |
| `Qwen/Qwen3-1.7B` | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | 1.7 | `Qwen/Qwen3-4B` |
| `Qwen/Qwen3-4B` | `1cfa9a7208912126459214e8b04321603b3df60c` | 4.0 | `Qwen/Qwen3-8B` |
| `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` | 8.0 | None |

`render_prompt` has two paths:
- **Chat path,** when `tokenizer.chat_template` is set: `apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)`. `enable_thinking=False` turns off Qwen3's reasoning mode; other templates ignore it.
- **Plain path:** an optional `f"{system}\n\n"`, then `f"### Question:\n{prompt}\n\n### Answer:\n"`.

`encode_sft` builds the token sequence as `ids = prompt_ids + response_ids + [eos]`, where `prompt_ids` is the tokenized `render_prompt(...)` without extra special tokens. Prompt tokens get the label `-100`. If the sequence is longer than `max_len`:
1. Cap the response (plus EOS) at `max_len - max_len // 4`, cutting from the right and keeping the EOS.
2. Left-truncate the prompt to fill the remaining space.
3. Set `truncated=True`.

- [ ] **Step 1: Write the failing tests**

```python
def test_registry_qwen3():
    i = model_info("Qwen/Qwen3-0.6B")
    assert i.reference_id == "Qwen/Qwen3-1.7B" and i.unseen_after == date(2025, 5, 1)
    assert i.revision == "c1899de289a04d12100db370d81485cdf75e47ca"
def test_reference_chain():  # 1.7B->4B, 4B->8B, 8B->None
def test_unknown_model():  # family "unknown"; reference_id, unseen_after, revision are None
def test_pick_device():  # result in {"cuda","mps","cpu"}
def test_load_tiny(tiny):  # tokenizer.pad_token_id is not None; model.dtype == torch.float32
def test_render_plain(tiny):
    assert render_prompt(tok, "What?") == "### Question:\nWhat?\n\n### Answer:\n"
    assert render_prompt(tok, "What?", system="S").startswith("S\n\n")
def test_render_chat(tiny):
    tok.chat_template = ("{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}\n{% endfor %}"
                         "{% if add_generation_prompt %}<|assistant|>{% endif %}")
    assert render_prompt(tok, "Q") == "<|user|>Q\n<|assistant|>"
def test_encode_masks_prompt(tiny):
    # labels == -100 over the prompt, labels == input_ids after it, input_ids[-1] == eos, truncated False
def test_encode_truncates_long_prompt(tiny):
    # prompt of 2000 words, short response, max_len 64 -> len(input_ids) <= 64,
    # truncated True, full response still labeled
def test_encode_truncates_long_response(tiny):
    # response of 2000 words, max_len 64 -> labeled tokens <= 48, last id eos, prompt tokens kept
```

- [ ] **Step 2:** Run `uv run pytest tests/test_models.py tests/test_prompting.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement both modules.
- [ ] **Step 4:** Run the same command. Expected: PASS. The first run downloads the tiny model.
- [ ] **Step 5:** Commit: `feat: model registry, loading, SFT encoding`.

---

### Task 5: Per-example losses and the novelty buckets

**Files:**
- Create: `src/dvx/scan/losses.py`, `src/dvx/scan/novelty.py`, `tests/test_losses.py`, `tests/test_novelty.py`

**Interfaces:**
- Consumes: `encode_sft`, `Example`, `ScanConfig`.
- Produces from `losses.py`:
  - `response_losses(model, tokenizer, examples, max_len=512, batch_size=8) -> list[float]`: mean NLL in nats per response token. Batches are right-padded, under `torch.no_grad()`.
  - `token_logprobs(model, tokenizer, texts: list[str], max_len=1024, batch_size=8) -> list[np.ndarray]`: log-probs of tokens 1..n-1 of each text.
- Produces from `novelty.py`:
  - `NoveltyResult(status: Literal["ok","unavailable"], reason: str | None, labels: list[str], counts: dict[str,int], fractions: dict[str,float], noise_threshold: float | None, known_margin: float)`, with `.summary() -> dict` (every field except `labels`).
  - `mismatched_pairs(examples, n: int, seed: int) -> list[Example]`.
  - `classify(target: list[float], reference: list[float], mismatched_reference: list[float], known_margin: float, noise_percentile: float) -> NoveltyResult`.
  - `novelty_unavailable(reason: str, n: int) -> NoveltyResult`.

`mismatched_pairs`: take a seeded sample of `m = min(n, len)` examples and pair the prompt of the i-th with the response of the `(i + 1) % m`-th. Raise `ValueError` if `len < 2`.

The classification rule, based on RHO-LOSS (spec §5.1):

```
noise_threshold = np.percentile(mismatched_reference, noise_percentile)
label = "noise"     if reference >= noise_threshold
        "known"     elif target - reference <= known_margin
        "learnable" otherwise
fractions = counts / n
```

- [ ] **Step 1: Write the failing tests**

```python
def test_losses_positive(tiny):  # len matches the input, all > 0
def test_losses_batch_invariant(tiny):
    # batch_size=1 vs 3 on mixed lengths -> np.allclose(rtol=1e-4); padding must not leak
def test_token_logprobs(tiny):  # each array length == len(tokens) - 1, all <= 0
def test_mismatched_pairs():
    # 5 examples, n=3 -> 3 pairs, no pair keeps its own response; 1 example -> ValueError
def test_classify():
    r = classify(target=[0.5, 2.0, 3.0], reference=[0.45, 1.0, 3.5],
                 mismatched_reference=list(np.linspace(3.0, 5.0, 21)),
                 known_margin=0.1, noise_percentile=10.0)
    assert r.labels == ["known", "learnable", "noise"]
    assert r.noise_threshold == pytest.approx(3.2) and r.fractions["known"] == pytest.approx(1/3)
def test_unavailable():
    r = novelty_unavailable("no reference model", 4)
    assert r.status == "unavailable" and r.reason == "no reference model"
```

- [ ] **Step 2:** Run `uv run pytest tests/test_losses.py tests/test_novelty.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the same command. Expected: PASS.
- [ ] **Step 5:** Commit: `feat: per-example losses and novelty buckets`.

---

### Task 6: Exposure estimate and its calibration

**Files:**
- Create: `src/dvx/scan/exposure.py`, `src/dvx/scan/calibration.py`, `tests/test_exposure.py`, `tests/test_calibration.py`

**Interfaces:**
- Produces from `exposure.py`:
  - `min_k_score(logprobs: np.ndarray, k=0.2) -> float`: the mean of the lowest `ceil(k * n)` values, or `nan` when `n == 0`.
  - `auc(pos, neg) -> float`: Mann-Whitney AUC, with ties counted as 0.5.
  - `ExposureResult(status: Literal["ok","inconclusive"], reason: str | None, auc: float | None, threshold: float | None, seen_fraction: float | None, n_calibration: int)`.
  - `assess_exposure(dataset_scores, seen_scores, unseen_scores, min_auc=0.6) -> ExposureResult`.
  - `exposure_inconclusive(reason: str) -> ExposureResult`.
- Produces from `calibration.py`:
  - `class CalibrationUnavailable(RuntimeError)`.
  - `fetch_calibration_texts(unseen_after: date, n_per_side: int, cache_dir: Path, fetch_json: Callable[[dict], dict] = wikipedia_get) -> tuple[list[str], list[str]]`, returning `(seen, unseen)`.

`assess_exposure` rules:
- `auc(seen, unseen) < min_auc` gives `inconclusive` with reason `f"calibration AUC {auc:.2f} below {min_auc}"`.
- Otherwise the threshold maximizes TPR − FPR over the calibration scores, where "positive" means seen and score ≥ threshold.
- `seen_fraction` is the share of non-NaN dataset scores at or above the threshold.

Calibration source (spec §5.1). Requests go to `wikipedia_get`: GET `https://en.wikipedia.org/w/api.php` with a `User-Agent` header of `dvx/0.1 (https://github.com/EmilFabel04/dvx)`. Network errors become `CalibrationUnavailable`.
- **Titles:** use `list=logevents&letype=create&lenamespace=0&ledir=newer&lelimit=500` with these `lestart` dates:
  - Seen: 2016-03-01, 2017-03-01, 2018-03-01, 2019-03-01.
  - Unseen: `unseen_after` + 30, 90, 180 and 270 days, skipping future dates.
- **Intros:** `prop=extracts&exintro=1&explaintext=1`, 20 titles per call. Keep intros of at least 400 characters and truncate them to 1,000 characters. Matching lengths reduces confounds.
- **Quota:** take `n_per_side // len(starts)` from each start date. If either side ends with fewer than `n_per_side // 2` texts, raise `CalibrationUnavailable`.
- **Cache:** `cache_dir / f"wiki_{unseen_after.isoformat()}_{n_per_side}.json"`.

- [ ] **Step 1: Write the failing tests**

```python
def test_min_k():
    assert min_k_score(np.array([-1., -2., -3., -4., -5.]), k=0.4) == -4.5
    assert np.isnan(min_k_score(np.array([])))
def test_auc():  # auc([3,4],[1,2]) == 1.0; auc([1],[1]) == 0.5; auc([1,2],[3,4]) == 0.0
def test_inconclusive_low_auc():
    r = assess_exposure([0.0], [0, 1, 0, 1], [0, 1, 1, 0])
    assert r.status == "inconclusive" and "AUC" in r.reason
def test_ok():
    r = assess_exposure([-1.0, -3.0, -1.05, float("nan")],
                        [-1.0, -1.1, -0.9], [-3.0, -3.1, -2.9])
    assert r.status == "ok" and r.auc == 1.0 and r.seen_fraction == pytest.approx(2/3)
def test_fetch_uses_cache(tmp_path):
    # fake fetch_json answers logevents and extracts from canned dicts and counts calls.
    # A second identical call makes no new calls. Each returned text is 400..1000 chars.
def test_fetch_failure(tmp_path):
    # fake raises OSError -> CalibrationUnavailable
```

- [ ] **Step 2:** Run `uv run pytest tests/test_exposure.py tests/test_calibration.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the same command. Expected: PASS.
- [ ] **Step 5:** Commit: `feat: membership-inference exposure with Wikipedia calibration`.

---

### Task 7: Evaluation suite registry

**Files:**
- Create: `src/dvx/evals/__init__.py`, `src/dvx/evals/registry.py`, `tests/test_registry.py`

**Interfaces:**
- Produces:
  - `MCQItem(benchmark: str, item_id: str, question: str, options: tuple[str, ...], answer: int)`, frozen.
  - `Benchmark(name, group: Literal["target","adjacent","side_effect"], hf_id, config: str | None, split, revision, size: int, to_item: Callable[[dict, int], MCQItem])`.
  - `Suite(domain: str, version: str, benchmarks: tuple[Benchmark, ...])`, with `.by_group(group) -> list[Benchmark]`.
  - `SuiteItems = dict[str, list[MCQItem]]`, keyed by benchmark name.
  - `class UnknownDomain(ValueError)`.
  - `SUITES`.
  - `get_suite(domain) -> Suite`.
  - `load_items(bench, rows=None, seed=0) -> list[MCQItem]`. When `rows` is None, call `datasets.load_dataset(hf_id, config, split=split, revision=revision)`. If there are more than `size` items, take a `random.Random(seed).sample` of `size` from the list sorted by `item_id`. Return the result sorted by `item_id`.
  - `load_suite_items(suite) -> SuiteItems`.

The `medical-qa` suite, version `"1"`:

| Group | Name | Source | Split | Size |
|---|---|---|---|---|
| target | `medqa` | `GBaker/MedQA-USMLE-4-options` @ `0fb93dd23a7339b6dcd27e241cb9b5eca62d4d18` | test | 500 |
| target | `medmcqa` | `openlifescienceai/medmcqa` @ `91c6572c454088bf71b679ad90aa8dffcd0d5868` | validation | 500 |
| target | `pubmedqa` | `qiaojin/PubMedQA` config `pqa_labeled` @ `9001f2853fb87cab8d220904e0de81ac6973b318` | train | 500 |
| adjacent | `mmlu_college_biology`, `mmlu_high_school_biology`, `mmlu_college_chemistry`, `mmlu_high_school_chemistry` | `cais/mmlu`, config = subject, @ `c30699e8356da336a370243923dbaf21066bb9fe` | test | 200 each |
| side_effect | `arc_challenge` | `allenai/ai2_arc` config `ARC-Challenge` @ `210d026faf9955653af8916fad021475a3f00453` | test | 500 |
| side_effect | `mmlu_high_school_mathematics`, `mmlu_philosophy`, `mmlu_high_school_macroeconomics`, `mmlu_miscellaneous` | `cais/mmlu`, as above | test | 200 each |

Converters. `idx` is the row index.

| Benchmark | `item_id` | `options` | `answer` / question |
|---|---|---|---|
| medqa | `f"medqa-{idx}"` | values of `options` dict in A–D order | `"ABCD".index(answer_idx)` |
| medmcqa | `row["id"]` | `(opa, opb, opc, opd)` | `cop` |
| pubmedqa | `f"pubmedqa-{pubid}"` | `("yes","no","maybe")` | question is `"Context: " + " ".join(context["contexts"]) + "\n\nQuestion: " + question`; answer is the index of `final_decision` |
| mmlu | `f"mmlu-{subject}-{idx}"` | `choices` | `answer` |
| arc | `row["id"]` | `choices["text"]` | `choices["label"].index(answerKey)` |

- [ ] **Step 1: Write the failing tests**

```python
def test_medical_suite():
    s = get_suite("medical-qa")
    assert s.version == "1"
    assert [b.name for b in s.by_group("target")] == ["medqa", "medmcqa", "pubmedqa"]
    assert [b.name for b in s.by_group("side_effect")][0] == "arc_challenge"
    assert len(s.by_group("adjacent")) == 4 and len(s.by_group("side_effect")) == 5
def test_unknown_domain():  # get_suite("law") raises UnknownDomain, message contains "medical-qa"
@pytest.mark.parametrize("name,row,expected", [...])  # one canned row per converter, incl. arc label "3" -> 2
def test_converters(name, row, expected):
def test_load_items_sampling():
    # 10 canned rows, size 3 -> 3 items, sorted by item_id; same seed equal; 2 rows -> 2 items
@pytest.mark.slow
def test_load_medqa_from_hub():  # 500 items, each with 4 options
```

- [ ] **Step 2:** Run `uv run pytest tests/test_registry.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the same command. Expected: PASS. Then run `uv run pytest tests/test_registry.py -m slow -v` once locally. Expected: PASS.
- [ ] **Step 5:** Commit: `feat: medical-qa evaluation suite registry`.

---

### Task 8: Multiple-choice letter scorer

**Files:**
- Create: `src/dvx/evals/scorer.py`, `tests/test_scorer.py`
- Modify: `src/dvx/prompting.py` (add `render_mcq`)

**Interfaces:**
- Consumes: `MCQItem`, `render_prompt`'s chat/plain switch.
- Produces:
  - `render_mcq(tokenizer, item: MCQItem) -> str`.
  - `letter_token_ids(tokenizer, n: int) -> list[list[int]]`.
  - `score_mcq(model, tokenizer, items, batch_size=8, max_len=1024) -> np.ndarray[bool]`.

`render_mcq` builds the user content as `f"{question}\n\n" + "\n".join(f"{L}. {opt}") + "\n\nAnswer with the letter of the correct option."`.
- Chat path: wrap it as a single user turn, using the same `apply_chat_template` call as `render_prompt`.
- Plain path: the content followed by `"\nAnswer:"`.

`letter_token_ids`: for each letter L in `"ABCDEFGH"[:n]`, collect the single-token encodings of `"L"` and `" L"`, skipping any multi-token encoding.

`score_mcq`:
- Tokenize with left padding, truncating from the left to `max_len`.
- Take the next-token logits at the last position.
- A letter's score is the maximum logit over its ids. The prediction is the argmax over letters.

- [ ] **Step 1: Write the failing tests**

```python
def test_render_mcq_plain(tiny):
    t = render_mcq(tok, MCQItem("b", "1", "Q?", ("x", "y"), 1))
    assert "A. x\nB. y" in t and t.endswith("Answer:")
def test_letter_ids(tiny):  # 4 lists, each non-empty, pairwise disjoint
def test_scoring_uses_last_logits(tiny):
    # StubLM(nn.Module) returns SimpleNamespace(logits=...) with a high logit for letter B's
    # ids at the last position. Items of mixed lengths: answer 1 -> True, answer 0 -> False.
def test_real_tiny_deterministic(tiny):  # bool array of len n; identical on a second call
```

- [ ] **Step 2:** Run `uv run pytest tests/test_scorer.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the same command. Expected: PASS.
- [ ] **Step 5:** Commit: `feat: log-likelihood letter scoring for multiple-choice evals`.

---

### Task 9: Statistics (bootstrap CIs, deltas, baseline check)

**Files:**
- Create: `src/dvx/evals/stats.py`, `tests/test_stats.py`

**Interfaces:**
- Produces:
  - `Estimate(value: float, ci_low: float, ci_high: float)`, frozen, with `.includes_zero() -> bool`.
  - `group_accuracy(correct: dict[str, np.ndarray], n_boot: int, seed: int) -> Estimate`: macro-average accuracy in percent.
  - `paired_delta(a: dict[str, np.ndarray], b: dict[str, np.ndarray], n_boot: int, seed: int) -> Estimate`: macro-average of (b − a) in percentage points.
  - `BaselineCheck(ok: bool, margin_pp: float, per_benchmark_pp: dict[str, float], required_pp: float)`.
  - `baseline_check(correct: dict[str, np.ndarray], n_options: dict[str, int], min_margin: float) -> BaselineCheck`.

Bootstrap method:
- Each replicate resamples item indices independently within each benchmark, using `np.random.default_rng(seed)`.
- `paired_delta` applies the same indices to `a` and `b`.
- The CI is the 2.5th to 97.5th percentiles.

Baseline: margin = mean over benchmarks of (accuracy − 1/k), and `ok = margin >= min_margin`.

- [ ] **Step 1: Write the failing tests**

```python
def test_group_accuracy():
    e = group_accuracy({"a": np.array([1, 1, 0, 0], bool)}, 200, 0)
    assert e.value == 50.0 and e.ci_low <= 50.0 <= e.ci_high
def test_paired_delta_macro():
    a = {"x": np.zeros(4, bool), "y": np.ones(4, bool)}
    b = {"x": np.array([1, 1, 0, 0], bool), "y": np.ones(4, bool)}
    assert paired_delta(a, b, 200, 0).value == 25.0
def test_identical_zero():  # value 0.0, ci (0.0, 0.0)
def test_ci_ordered():  # random data -> ci_low <= value <= ci_high
def test_baseline():
    # 4-option at 30% and 3-option at 40%: margin ≈ 5.83 pp, ok with 0.05; fails with 0.10
def test_includes_zero():
    assert Estimate(1.0, -0.5, 2.0).includes_zero() and not Estimate(1.0, 0.2, 2.0).includes_zero()
```

- [ ] **Step 2:** Run `uv run pytest tests/test_stats.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the same command. Expected: PASS.
- [ ] **Step 5:** Commit: `feat: bootstrap CIs, paired deltas, baseline check`.

---

### Task 10: Contamination detection

**Files:**
- Create: `src/dvx/contamination.py`, `tests/test_contamination.py`

**Interfaces:**
- Consumes: `Example`, `MCQItem`, `normalize_text`.
- Produces:
  - `ContaminationResult(n_flagged: int, flagged_lines: list[int], by_benchmark: dict[str, int], ngram: int)`.
  - `find_contamination(examples, items, n=13) -> ContaminationResult`.

Matching rules:
- Words come from `normalize_text`, with punctuation stripped.
- An item's text is `question + " " + " ".join(options)`.
- Items with at least `n` words are matched on any shared word n-gram with the example's `prompt + " " + response`.
- Shorter items are matched when their normalized text is a substring of the normalized example.
- An example counts once per benchmark it hits.

- [ ] **Step 1: Write the failing tests**

```python
def test_flags_shared_13gram():
    # 20-word medqa question; an example prompt contains 13 consecutive words from it with
    # different case and punctuation -> flagged_lines == [1], by_benchmark == {"medqa": 1}
def test_short_item_substring():  # 6-word item fully inside the example -> flagged
def test_clean():  # unrelated texts -> n_flagged == 0
```

- [ ] **Step 2:** Run `uv run pytest tests/test_contamination.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the same command. Expected: PASS.
- [ ] **Step 5:** Commit: `feat: n-gram benchmark contamination detection`.

---

### Task 11: LoRA fine-tuning

**Files:**
- Create: `src/dvx/train.py`, `tests/test_train.py`

**Interfaces:**
- Consumes: `TrainConfig`, `encode_sft`, `count_params`.
- Produces:
  - `TrainStats(n_examples, n_truncated, tokens_trained, steps, wall_seconds, est_flops, final_loss, seed)`.
  - `finetune_lora(model, tokenizer, examples, cfg: TrainConfig, seed: int) -> tuple[PeftModel, TrainStats]`.

The training loop:
- Set seeds for `random`, `numpy` and `torch`.
- Read `n_params = count_params(model)` *before* wrapping it with `peft.get_peft_model` (LoRA config from `cfg`, `task_type="CAUSAL_LM"`).
- Encode every example, then shuffle with `random.Random(seed)`.
- Right-pad micro-batches of `cfg.micro_batch_size` and accumulate gradients over `cfg.grad_accum` micro-batches.
- Optimizer: AdamW over trainable parameters only. Learning rate: linear warmup over `ceil(warmup_ratio * steps)` steps, then linear decay to 0.
- `steps = ceil(n_examples * epochs / (micro_batch_size * grad_accum))`.
- `tokens_trained` = non-pad input tokens processed.
- `est_flops = 6 * n_params * tokens_trained`, the 6·N·D approximation, documented in the docstring.
- `final_loss` = mean loss of the last optimizer step.
- Return the model in eval mode.

- [ ] **Step 1: Write the failing tests** (CPU, a fresh tiny model per test)

```python
def test_loss_decreases():
    exs = [Example("Q", "the answer is blue", None, i) for i in range(16)]
    cfg = TrainConfig(lr=1e-2, epochs=3, max_seq_len=64)
    before = response_losses(m, tok, exs[:1], max_len=64)[0]
    m2, st = finetune_lora(m, tok, exs, cfg, seed=0)
    assert response_losses(m2, tok, exs[:1], max_len=64)[0] < before
    assert st.n_examples == 16 and st.steps == 6
def test_tokens_and_flops():
    # tokens_trained == 3 * sum(len(encode_sft(...).input_ids));
    # est_flops == 6 * count_params(base) * tokens_trained
def test_seed_determinism():  # two fresh runs with seed 0 -> equal final_loss
def test_counts_truncated():  # one example longer than max_seq_len -> n_truncated == 1
```

- [ ] **Step 2:** Run `uv run pytest tests/test_train.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the same command. Expected: PASS.
- [ ] **Step 5:** Commit: `feat: LoRA fine-tuning with token and FLOP accounting`.

---

### Task 12: Run record and corpus

**Files:**
- Create: `src/dvx/record.py`, `tests/test_record.py`

**Interfaces:**
- Produces:
  - `SCHEMA_VERSION = 1`.
  - `RunRecord(schema_version, run_id, created_at: str, status: Literal["complete","stopped_baseline","scan_only"], engine_version, profile, domain, suite_version, env: dict, dataset: dict, model: dict, seeds: list[int], config: dict, results: dict, compute: dict)`, a dataclass.
  - `new_run_id(now: datetime, fingerprint: str) -> str`.
  - `env_info(device: str) -> dict`, with keys `python, platform, torch, transformers, peft, datasets, device, dvx`.
  - `save_record(record, run_dir) -> Path` (writes `record.json`).
  - `append_to_corpus(record, corpus_path) -> None`. Appends one JSON line and raises `ValueError` if the `run_id` is already present.
  - `read_corpus(corpus_path) -> list[dict]`.

These dict layouts are pinned because the report reads them:
- **`dataset`:** `path, fingerprint, n_total, n_scanned, n_trained, declared_type, declared_date_range`.
- **`model`:** `id, revision, params, family, reference_id, unseen_after`.
- **`results`**, all keys optional depending on status:
  - `duplicates` (DuplicateStats)
  - `contamination` (ContaminationResult)
  - `novelty` (`NoveltyResult.summary()`)
  - `exposure` (ExposureResult)
  - `baseline` (BaselineCheck)
  - `impact`, containing:
    - `accuracy.{base,control,tuned}.{target,adjacent,side_effect}` (Estimate)
    - `per_benchmark_accuracy.{base,control,tuned}.{bench}` (float, percent)
    - `delta_data_specific.{group}` (tuned − control)
    - `delta_raw.{group}` (tuned − base)
    - `delta_control.{group}` (control − base)
    - `train.{tuned,control}` (TrainStats)
- **`compute`:** `steps.{name}.wall_seconds`, `tokens_trained_total`, `est_flops_total`.

All dataclasses are stored through `dataclasses.asdict`.

- [ ] **Step 1: Write the failing tests**

```python
def test_run_id():
    assert new_run_id(datetime(2026, 10, 30, 21, 5, 9), "abcdef0123" + "0" * 54) \
        == "20261030-210509-abcdef01"
def test_roundtrip(tmp_path):
    # save_record then json.load == asdict(rec); append twice -> ValueError; read_corpus -> 1 row
def test_env_info():
    assert set(env_info("cpu")) == {"python", "platform", "torch", "transformers", "peft",
                                    "datasets", "device", "dvx"}
```

- [ ] **Step 2:** Run `uv run pytest tests/test_record.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the same command. Expected: PASS.
- [ ] **Step 5:** Commit: `feat: run record schema and append-only corpus`.

---

### Task 13: Report rendering

**Files:**
- Create: `src/dvx/report.py`, `tests/test_report.py`, `tests/fixtures.py` (`sample_record(**overrides) -> dict`, a complete quick-run record)

**Interfaces:**
- Consumes: the record dict layout from Task 12.
- Produces:
  - `render_report(record: dict) -> str`, in Markdown.
  - `render_summary(record: dict) -> str`, for the terminal: header, headline, side-effect line and warnings.
  - `fmt_delta(e: dict) -> str`.

Exact wording. Tests assert these strings.
- **`fmt_delta`:** `"+3.2 pp (95% CI +1.0 to +5.4)"`. If the CI includes zero, append `" — not distinguishable from zero"`.
- **Header:** `f"# Data Impact Report: {model.id} on {dataset.path}"`, then a context line with domain, suite version, profile, the fingerprint's first 12 characters, model revision, parameters and seeds.
- **Sample notes:**
  - When `n_scanned < n_total`: `f"Measured on a sample of {n_scanned:,} of {n_total:,} examples."`
  - When `n_trained < n_total`: `f"Impact measured by training on {n_trained:,} of {n_total:,} examples."`
- **Headline:** `"Data-specific gain on target benchmarks: " + fmt_delta(...)`.
- **"Side effects" section:** a table of `delta_data_specific` and `delta_raw` for all three groups, placed directly after the headline. Regressions are not shown below gains.
- **Per-benchmark table:** base, control and tuned accuracy.
- **Novelty:**
  - If ok: fractions as percentages for known, learnable and noise.
  - If unavailable: `f"Novelty: unavailable ({reason})"`.
- **Exposure:**
  - If ok: `f"About {seen_fraction:.0%} of examples look previously seen by this model (calibration AUC {auc:.2f})."`, plus the caveat line `"Calibration compares old vs. new Wikipedia articles; topic drift can inflate the AUC."`
  - If inconclusive: `f"Prior exposure: inconclusive ({reason})."`
- **Duplicates:** exact and near percentages.
- **Truncation:** when `impact.train.tuned.n_truncated > 0`, `f"{n_truncated} training examples were truncated to {max_seq_len} tokens."` Take `max_seq_len` from `config.train.max_seq_len`.
- **Contamination:** when `n_flagged > 0`, `f"WARNING: {n_flagged} examples overlap with evaluation items ({by_benchmark}). Gains on those benchmarks may be inflated."`
- **Baseline stop:** for `status == "stopped_baseline"`, `f"Stopped: the base model is not clearly above chance on the target benchmarks (margin {margin_pp:.1f} pp, need {required_pp:.1f} pp). Try a larger model."`, with no impact sections.
- **Caveats section, always present:**
  - `"Quick profile: one training seed; confidence intervals cover question sampling only, not training randomness."`
  - `"The control is generic instruction data; part of the raw gain can come from learning the answer format."`

- [ ] **Step 1: Write the failing tests:** one test per bullet above, each building `sample_record(...)` with the relevant override and asserting the exact substring. Also `test_fmt_delta_positive_and_zero` and `test_summary_has_headline_and_warnings`.
- [ ] **Step 2:** Run `uv run pytest tests/test_report.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the same command. Expected: PASS.
- [ ] **Step 5:** Commit: `feat: Markdown report and terminal summary`.

---

### Task 14: Step cache and pipeline (scan, quick run, resume)

**Files:**
- Create: `src/dvx/run/__init__.py`, `src/dvx/run/steps.py`, `src/dvx/run/pipeline.py`, `tests/test_steps.py`, `tests/test_pipeline.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `cached_step(run_dir: Path, name: str, fn: Callable[[], dict]) -> dict`. If `steps/{name}.json` exists, load and return it. Otherwise run `fn`, add `"_wall_seconds"`, write to `{name}.json.tmp`, then `os.replace` it into place.
  - `RunRequest(data_path: Path, model_id: str, revision: str | None, domain: str, profile: str, reference_id: str | None, unseen_after: date | None, declared_type: str | None, declared_date_range: tuple[str, str] | None, home: Path)`, frozen. It's serialized to `runs/<id>/request.json`.
  - `start_run(req, *, profile_obj: Profile | None = None, items: SuiteItems | None = None, calibration_fetcher=fetch_calibration_texts, device: str | None = None, now: datetime | None = None) -> RunRecord`.
  - `resume_run(run_id: str, home: Path, **same keyword overrides) -> RunRecord`.
  - `run_scan(req, **same keyword overrides) -> RunRecord`, with `status="scan_only"`.

Resolving defaults:
- `revision`, `reference_id` and `unseen_after` fall back to `model_info(model_id)`.
- `profile_obj` defaults to `PROFILES[req.profile]`.
- `items` default to `load_suite_items(get_suite(req.domain))`.
- `device` defaults to `pick_device()`.

Steps, in this order and with these exact names. Each loads the models it needs and calls `release()` before returning.
1. **`dataset`:** load, fingerprint, take the scan sample (`measure_cap`, seed 0) and the train sample (`train_cap`, seed 0, drawn from the scan sample).
2. **`duplicates`:** over all examples.
3. **`contamination`:** the train sample against all suite items.
4. **`scan_target`:**
   - `response_losses` on the scan sample;
   - `min_k_score` of `token_logprobs` for the dataset texts (`prompt + "\n" + response`);
   - the same scores for the calibration texts, when a fetch succeeded.

   This step also stores `n_params = count_params(model)`, which becomes `model.params` in the record. The calibration fetch happens inside this step. If `unseen_after` is None, the reason is `"model release date unknown; pass --unseen-after"`. If `CalibrationUnavailable` is raised, the reason is `"calibration unavailable: {err}"`.
5. **`scan_reference`:** `response_losses` on the scan sample and on `mismatched_pairs(scan, mismatched_n, seed=0)`. Skipped, with the reason `"no reference model"`, when `reference_id` is None or there are fewer than 2 examples.
6. **`eval_base`:** `score_mcq` per benchmark, stored as lists of bools. The baseline check then runs here. On failure: status `stopped_baseline`, then save and report.
7. **`tuned_s0`:** fresh target, `finetune_lora` on the train sample, then evaluate.
8. **`control_s0`:** fresh target, `finetune_lora` on `load_control_examples(len(train_sample), seed=0)`, then evaluate.
9. **Assemble:** novelty from `classify`, exposure from `assess_exposure`, the stats, then `save_record`, `append_to_corpus` (`home / "corpus.jsonl"`) and `report.md`.

`run_scan` covers steps 1–5 plus assembly.

Tests use `device="cpu"`, `TINY` as both target and reference, tiny `items` (2 per benchmark name of `medical-qa`), a fake calibration fetcher, and a profile override of `replace(PROFILES["quick"], train_cap=8, baseline_margin=-1.0)`.

- [ ] **Step 1: Write the failing tests**

```python
# test_steps.py
def test_cached_step_runs_once(tmp_path):  # second call doesn't invoke fn; result has "_wall_seconds"
def test_partial_tmp_ignored(tmp_path):  # a leftover steps/x.json.tmp -> fn still runs
# test_pipeline.py (12-example fixture JSONL)
def test_quick_end_to_end(tmp_path):
    rec = start_run(req, profile_obj=P, items=ITEMS, calibration_fetcher=fake, device="cpu")
    assert rec.status == "complete"
    assert set(rec.results) == {"duplicates", "contamination", "novelty", "exposure",
                                "baseline", "impact"}
    assert set(rec.results["impact"]["delta_data_specific"]) == {"target", "adjacent",
                                                                  "side_effect"}
    # record.json and report.md exist; corpus.jsonl has 1 line
def test_baseline_stop(tmp_path):
    # baseline_margin=0.99 -> status "stopped_baseline"; no steps/tuned_s0.json; report has "Stopped:"
def test_resume_skips_completed(tmp_path, monkeypatch):
    # run; delete steps/control_s0.json, record.json and the corpus line;
    # count finetune_lora calls during resume_run == 1
def test_unknown_model_and_offline(tmp_path):
    # reference_id=None, fetcher raises CalibrationUnavailable -> complete;
    # novelty.status == "unavailable" with reason "no reference model";
    # exposure.status == "inconclusive" with reason starting "calibration unavailable"
def test_scan_only(tmp_path):  # run_scan -> status "scan_only", no "impact" key
def test_sample_counts(tmp_path):  # measure_cap=10 on 12 examples -> n_total 12, n_scanned 10
```

- [ ] **Step 2:** Run `uv run pytest tests/test_steps.py tests/test_pipeline.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `uv run pytest -v`. Expected: the whole suite passes.
- [ ] **Step 5:** Commit: `feat: step-cached pipeline with scan, quick run and resume`.

---

### Task 15: CLI

**Files:**
- Create: `src/dvx/cli.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `start_run`, `resume_run`, `run_scan`, `get_suite`, `read_corpus`, `render_report`, `render_summary`.
- Produces: `main(argv: list[str] | None = None) -> int`, built with argparse subcommands:
  - `scan DATA --model M [--revision R] [--reference R] [--domain medical-qa] [--unseen-after YYYY-MM-DD]`
  - `run DATA --model M --domain D --profile {quick} [--revision] [--reference] [--unseen-after] [--type T] [--date-range START END] [--device {cuda,mps,cpu}]`
  - `resume RUN_ID`
  - `report RUN_ID`, which re-renders `report.md` from `record.json` and prints the summary.
  - `corpus list`, which prints `run_id  status  model  dataset-fingerprint[:12]  headline`.
  - `corpus export OUT`.
  - Every subcommand accepts `--home`, defaulting to the `DVX_HOME` environment variable, else `.dvx`.

Behavior:
- `get_suite(domain)` is validated *before* any loading.
- `DatasetFormatError`, `UnknownDomain` and `FileNotFoundError` are printed to stderr as `f"dvx: error: {err}"` and return 2.
- Success prints `render_summary` and the report path, and returns 0.

- [ ] **Step 1: Write the failing tests**

```python
def test_help():  # main(["--help"]) raises SystemExit(0)
def test_bad_dataset_exits_2(tmp_path, capsys):
    # invalid first line -> 2; "line 1" in stderr; no traceback in stderr
def test_unknown_domain_exits_2(tmp_path, capsys):  # --domain law -> 2; "medical-qa" in stderr
def test_run_builds_request(tmp_path, monkeypatch):
    # monkeypatch dvx.cli.start_run to capture req -> return code 0;
    # req.declared_date_range == ("2020-01-01", "2023-12-31")
    # req.declared_type == "medical-qa"; req.revision is None
def test_corpus_list(tmp_path, capsys):  # one record in corpus -> its run_id printed
def test_report_rerender(tmp_path):
    # saved sample record -> main(["report", id, "--home", ...]) == 0 and report.md rewritten
```

- [ ] **Step 2:** Run `uv run pytest tests/test_cli.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `uv run pytest -v && uv run dvx --help`. Expected: everything passes and the help text lists the subcommands.
- [ ] **Step 5:** Commit: `feat: dvx CLI (scan, run, resume, report, corpus)`.

---

### Task 16: First real run on the M5 and timing record

**Files:**
- Create: `scripts/make_medmcqa_sample.py`, `tests/test_make_sample.py`, `docs/timings-m5.md`

**Interfaces:**
- Produces: `medmcqa_row_to_example(row: dict) -> dict` and a CLI `--n --seed --out`. It loads the `openlifescienceai/medmcqa` train split at the Task 7 revision and writes JSONL.
  - The prompt is the question, then a blank line, then `A.`–`D.` option lines, then a blank line, then `"Answer with the letter of the correct option."`
  - The response is `f"{L}. {option}"`, plus `f"\n\nExplanation: {exp}"` when `exp` is non-empty.

- [ ] **Step 1: Write the failing test.** `test_row_to_example`: a canned row with `cop=2` and `exp="because"` gives a response starting with `"C. "` and containing `"Explanation: because"`. A row with `exp=None` has no `"Explanation"`.
- [ ] **Step 2:** Run `uv run pytest tests/test_make_sample.py -v`. Expected: FAIL. Then implement the script. Expected: PASS.
- [ ] **Step 3:** Run `uv run python scripts/make_medmcqa_sample.py --n 2000 --seed 0 --out data/medmcqa_train_2k.jsonl`. Expected: 2000 lines.
- [ ] **Step 4:** Run `uv run dvx run data/medmcqa_train_2k.jsonl --model Qwen/Qwen3-0.6B --domain medical-qa --profile quick --type medical-qa`. Expected: it finishes with a summary and a report path.
  - If it stops at the baseline, record the margin and repeat with `--model Qwen/Qwen3-1.7B`.
  - The report's contamination line shows whether MedMCQA train overlaps the validation items. Whatever it says, record it.
- [ ] **Step 5:** Write `docs/timings-m5.md`, with one table per model run:
  - per-step `wall_seconds` from `record.json`;
  - tokens trained and peak memory, observed with Activity Monitor;
  - the baseline margin, the headline delta and the contamination count;
  - a projected `full`-profile duration, computed as tuned/control step time × 19 fine-tunes plus evals.

  M2's plan is sized from this file.
- [ ] **Step 6:** Commit the script, test and doc (`data/` is gitignored): `docs: first M5 quick-profile run and timings`.
