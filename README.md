# dvx

**Measure what a text dataset does to a language model.**

dvx takes **one dataset and one target model** and writes a **Data Impact Report**: what fine-tuning on that data changes in that model, against a generic-data control, with confidence intervals. Every number comes with the model, revision and context it was measured in. Each run also appends a structured record to a local **corpus**, the start of a dataset of "data about data value".

> **Status: Milestone 1 (the quick report).** It works end to end on a laptop. The full report (scaling curve, clean-gain rerun, cluster contributions, multiple seeds) is Milestone 2. The design is in [`docs/superpowers/specs/2026-10-30-data-impact-engine-design.md`](docs/superpowers/specs/2026-10-30-data-impact-engine-design.md).

## Install

You need Python 3.12 or newer and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/EmilFabel04/dvx.git
cd dvx
uv sync
```

Models, benchmarks and the control dataset come from the Hugging Face Hub on first use and are cached there afterwards.

## A first run

Make a dataset: 2,000 examples from the MedMCQA train split, in dvx's JSONL format. (MedMCQA's eval items come from its validation split.)

```bash
uv run python scripts/make_medmcqa_sample.py --n 2000 --seed 0 --out data/medmcqa_train_2k.jsonl
```

Scan it (part 1 only, no training):

```bash
uv run dvx scan data/medmcqa_train_2k.jsonl --model Qwen/Qwen3-1.7B-Base
```

Measure its impact with the quick profile (part 1, then part 2):

```bash
uv run dvx run data/medmcqa_train_2k.jsonl --model Qwen/Qwen3-1.7B-Base --domain medical-qa \
    --profile quick --type medical-qa
```

Each step prints a progress line to stderr and saves its output under `.dvx/runs/<run_id>/steps/`. A stopped run continues from its last finished step:

```bash
uv run dvx resume 20261101-201145-8c7ff65f
```

Re-render a finished run's report, and inspect the corpus:

```bash
uv run dvx report 20261101-201145-8c7ff65f
uv run dvx corpus list
uv run dvx corpus export corpus-export.jsonl
```

Runs and the corpus live in `.dvx/` (override it with `--home` or `$DVX_HOME`). Other options for `scan` and `run`:

- `--revision` pins the model revision;
- `--reference` picks the reference model for the novelty check;
- `--unseen-after YYYY-MM-DD` sets the date after which text is unseen by the model.

These are for `run` only:

- `--date-range START END` records when the data was written;
- `--device cuda|mps|cpu` overrides the device.

### Input format

One JSON object per line, either `{"prompt": ..., "response": ...}` (optionally with `"system"`) or a single-turn `{"messages": [...]}` with an optional system message, then one user and one assistant message. Every model sees the same plain-text format: the system text and the prompt joined by a blank line, a blank line, then the response. Chat templates are not used.

### Models

The registry pins the pretrained (Base) Qwen3 checkpoints, Apache-2.0: `Qwen/Qwen3-0.6B-Base`, `Qwen/Qwen3-1.7B-Base`, `Qwen/Qwen3-4B-Base` and `Qwen/Qwen3-8B-Base`. The novelty reference for 0.6B, 1.7B and 4B is the next size up; 8B has none, so its novelty is reported as unavailable unless you pass `--reference`. Other Hub models work too: pass `--unseen-after` and `--reference`, and dvx records the commits it loaded. Post-trained (chat) checkpoints are not registered, because their losses reflect answer style rather than content.

## What the quick report contains

**Part 1: data checks (no training).** Novelty, prior exposure and contamination run on up to 20,000 examples; above that, a fixed random sample is used and the report says so. Duplicates are checked over the whole dataset.

- **Novelty.** Each example's response loss under the target model and under a larger reference model splits the data into *known*, *learnable* and *noise*. A reliability gate marks the split **inconclusive** when the reference's loss is below the target's on fewer than half the examples, or when the reference cannot tell true prompt/response pairs from mismatched ones (paired discrimination below 0.6). The report always shows both reliability numbers, the thresholds, loss quartiles and example items.
- **Prior exposure.** A Min-K% Prob membership score is calibrated on Wikipedia articles from before and after the model's cutoff. The estimate comes with its calibration AUC and is **inconclusive** below 0.6.
- **Duplicates.** Exact and MinHash near-duplicates.
- **Contamination.** Examples that contain an evaluation item (13-gram coverage and question phrases, robust to "A." / "(B)" option labels) are flagged, and the warning sits next to the gains.

**Part 2: impact (quick profile).**

- **Base model check.** Before any training, the base model must beat chance by at least 5 percentage points on average on the target benchmarks. Chance is the larger of 1/k and the most common answer's share. Otherwise the run stops and recommends a larger model.
- **Fine-tunes.** One LoRA fine-tune on up to 2,000 of your examples, and one on the same number of generic instruction examples (Databricks Dolly 15k, the control).
- **Evaluation.** Base, control and tuned models are scored on the pinned `medical-qa` suite (v2):
  - target: MedQA, MedMCQA and PubMedQA;
  - adjacent: four MMLU biology and chemistry subjects;
  - side effects: ARC-Challenge and four other MMLU subjects.
  Scoring is plain-text 5-shot log-likelihood. The shots come from rows disjoint from the eval items, and their IDs are recorded.
- **The headline** is the **data-specific gain** (tuned − control) on the target group, in percentage points with a 95% bootstrap confidence interval. Raw gain (tuned − base) and the format effect (control − base) are shown next to it. A group whose interval lies wholly below zero is called out as a regression, right after the headline.

**Not in the quick report (Milestone 2):**

- multiple training seeds (the `full` profile);
- the scaling curve (part 3);
- the clean-gain rerun without contaminated examples, and cluster contributions (part 4);
- the evaluation of larger sibling models for the conversion anchor.

There is no price or currency output.

## How far to trust a result

- **Uncertainty.** The quick profile trains **one seed**. Its confidence intervals cover question sampling only, not training randomness, so treat a small gain with a narrow interval with care.
- **Reproducibility.** Each record pins what a third party needs to rerun the measurement: the dataset content hash, model and reference revisions, the eval suite version (which pins benchmark revisions) and shot IDs, seeds, library versions, the dvx version and source commit (with a dirty flag), the calibration texts' SHA-256 and fetch date, and the hardware. Reruns are not bit-identical, though. Training and scoring on Apple GPUs (MPS) are not deterministic, the calibration texts are re-fetched from a changing Wikipedia unless the cache is kept, and a different device or library version can move the numbers. Compare reruns by their confidence intervals, not their digits.
- **Small models.** A result for a 1.7B model is not evidence about an 8B model.
- **Novelty buckets.** In M1 the noise threshold compares absolute loss, so long responses lean toward "noise". Treat the known/learnable/noise mix as indicative; the gate's reliability numbers say whether the reference model is usable, not whether the bucket boundaries are right. Milestone 2 redesigns the rule.

## Running on a 16 GB Apple Silicon Mac

dvx runs on MPS, CUDA and CPU from one codebase, loads one model at a time, and bounds scoring and training memory. Measured wall times and memory for a quick run on an M5 with 16 GB are in [`docs/timings-m5.md`](docs/timings-m5.md). A quick run with Qwen3-1.7B-Base took 8,011 s of step wall time (2 h 13 min), and the highest process footprint sampled every 30 s was 10,240 MB. One-time costs include downloading the models and fetching the Wikipedia calibration texts (about 20 minutes the first time).

## Licenses

dvx is [Apache-2.0](LICENSE). It redistributes no data or weights. Everything below is downloaded at runtime and used under its own license:

- the control set, Databricks Dolly 15k, is CC BY-SA 3.0;
- the benchmarks (MedQA, MedMCQA, PubMedQA, MMLU and ARC) are under their own licenses;
- the Qwen3 models are Apache-2.0;
- the calibration texts are Wikipedia article intros, CC BY-SA, cached locally.

Check each license before you redistribute anything a run produces.

## Development

```bash
uv sync --extra dev
uv run pytest            # fast tests; a tiny random model on CPU
uv run pytest -m slow    # real models and Hub downloads
uv run ruff check .
```
