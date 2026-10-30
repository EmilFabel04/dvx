# Data Impact Engine: v1 Design

*Status: draft for review, 2026-10-30. The working name is `dvx`, a placeholder.*

## 1. Intent and why

This project started from an idea for **data insurance**. You can't insure data without a credible way to value it, and today nobody measures that value.

How data is bought and sold today:

- **Catalogs** (Defined.ai, Datarade, AWS Data Exchange, Snowflake Marketplace) describe what a dataset *is*: its size, type, language, fields, license, and a free sample. They never say what it will *do* to a model.
- **Big licensing deals** (for example Reddit–Google, or news publishers with OpenAI) are priced by bargaining power and legal risk.
- **Commissioned data** from vendors like Scale, Surge and Mercor is priced by labor cost.

None of these is priced by measured impact. Buyers test samples themselves, negotiate, or guess. AI labs do have internal tools for measuring data value, but those tools are private, tied to the lab's own models, and not comparable across companies.

**What we're building** is an open, verifiable engine. It takes **one dataset and one target model** and produces a **Data Impact Report**: a breakdown of what that data does to that model, in that context. Anyone with the same inputs can rerun it and check the result. It is a tool people can verify, not an authority they have to trust.

**Why measure impact and not price:** you can measure impact; a price is a judgment. Measured impact is also the foundation that pricing and, later, insurance will need.

**Mission:** advance AI progress and the "data as currency" market. Anyone can use it: AI labs, data brokers, companies with heavy ML pipelines. Making money fast is not the goal.

**Long-term path** (context only; v1 builds just the first step):

1. Reports from the measurement engine.
2. Every report is saved into a growing collection of "data about data value" (the **corpus**).
3. A learned predictor trained on the corpus, which estimates impact from cheap signals without a training run.
4. A **conversion anchor** that turns impact into money, for example "this dataset gives the same gain as N GPU-hours of extra training", priced at market compute rates.
5. Pricing, and eventually data insurance.

## 2. Goals for v1

1. Given a **text dataset** in a domain with a registered evaluation suite (in v1, medical QA) and a **target open model**, produce a reproducible Data Impact Report in four parts (section 5).
2. Write one structured **corpus record** per run. It must be enough to reproduce the run, train the future predictor, and calculate the conversion anchor later.
3. Run locally on the owner's MacBook Pro (M5, 16 GB unified memory) for models up to about 3B parameters. The same code must run unchanged on a rented NVIDIA GPU, for validation on a larger (~8B) model.
4. Prove the engine works on the first test case, **medical question answering**, by passing the known-answer tests in section 8.

## 3. Non-goals for v1

- **Money.** No price or currency output. Record the ingredients for the conversion anchor (section 6) but don't calculate it.
- **Non-text data.** No images, audio, video or tabular data.
- **API, web UI or hosted service.** v1 is a CLI plus an importable Python library.
- **Scoring individual examples.** Methods like influence functions, TRAK or DataInf come later. v1 measures contributions at the cluster level.
- **The predictor model itself.** That needs a corpus first. v1 only guarantees the corpus records are good enough to train it.
- **Pretraining-scale experiments.** v1 measures the effect of *fine-tuning* (supervised instruction/QA tuning with LoRA adapters), not pretraining.
- **Privacy layers.** Clean rooms, confidential computing and seller/buyer privacy come later.
- **Marketplace or broker integrations.**
- **Domains without a registered evaluation suite.** v1 registers suites for medical QA only. Datasets in other domains get part 1 only, with a clear message saying so. Datasets whose value can't be measured by multiple-choice scoring (dialogue, code, creative writing) are out of scope. Users bringing their own evaluation files comes later. *[my call]*

## 4. Anti-goals: failure even if the code works

- **A single opaque "intelligence score"**, or any number shown without its model and context.
- **Results a third party can't reproduce** from the same inputs (dataset hash, model revision, seeds, engine version).
- **Gains reported without uncertainty**, or gains from benchmark-contaminated data reported as real.
- **Hidden side effects.** Regressions must appear in the report as prominently as gains.
- **Overclaiming transfer**: presenting small-model results as true for large models without the validation in section 8.
- **Presenting weak signals as facts.** The "has the model seen this?" estimate in particular (section 5.1) must carry its measured reliability.

## 5. The report: what each part computes

These four parts were chosen with the project owner. The methods inside each part are proposals marked *[my call]*. The builder may change a method if they keep the part's purpose.

### Inputs

- **Dataset:** a JSONL file of text examples in a prompt/response format, either `{"prompt": ..., "response": ...}` or chat `messages`. The builder may add format adapters.
- **Dataset size** *[my call]*. v1 measures datasets of up to about 20k examples in full; the builder confirms that limit from measured run times. For larger datasets:
  - exact-duplicate hashing still runs over the whole set;
  - part 1 and all training runs use a fixed random sample;
  - the report header says "measured on a sample of N of M examples";
  - anything beyond the measured range is labeled as extrapolation.

  Running whole large datasets in the cloud comes later.
- **Target model:** a Hugging Face model ID plus a pinned revision.
- **Context metadata**, which the owner named as factors in a dataset's value:
  - **Data type and domain** (for example "medical QA"). Declared by the user with `--domain`. The domain selects the evaluation suites from a per-domain **suite registry** with pinned suite versions, so results stay comparable across corpus records. v1 fills in only `medical-qa`. *[my call]*
  - **Data age.** The user declares the date range the data was collected or written. It's recorded and compared with the model's knowledge cutoff where that's known. v1 does *not* claim a rule for how age affects value; the corpus will reveal it later.
  - **Model size.** Read from the model and recorded.
  - **Prior exposure:** has the model seen this data before? *Measured* by part 1, not declared.

### 5.1 Part 1: Novelty scan (fast, no training)

Purpose: find out how much of this data the model doesn't already know.

- **Per-example surprise.** Measure the model's loss on each example's response, given its prompt. Low loss means the model already knows it.
- **Separating "learnable" from "noise"** *[my call]*. A high loss alone can't tell new-but-learnable data from junk. Use a **reference-model comparison**, the idea behind RHO-LOSS (Mindermann et al., 2022): also score each example with a larger reference model from the same family.
  - Hard for the target but easy for the reference: **learnable**.
  - Hard for both: **likely noise**.
  - Easy for the target: **already known**.
  The builder chooses thresholds and documents them.
- **Prior exposure estimate** *[my call]*. A membership-inference score such as Min-K% Prob (Shi et al., 2023) estimates whether the model saw these exact examples in training. **This signal is known to be weak.** Duan et al. (2024) found such attacks close to chance in many settings. The engine must therefore calibrate it against a control set of known-seen text (for example, widely trained-on public text) and known-unseen text (published after the model's cutoff). It reports the measured reliability (AUC). If reliability is near chance, the report says "inconclusive" instead of giving a number.
- **Duplicates.** Exact and near-duplicate detection (for example MinHash) within the dataset.
- **Output:** the share of examples that are known, learnable, noise and duplicate; the exposure estimate with its reliability; distributions; a few example items from each bucket.

### 5.2 Part 2: Impact test (short fine-tune)

Purpose: measure what the data actually changes in the model.

- **Fine-tuning.** LoRA fine-tune of the target model on the dataset with a fixed, documented default recipe. Training is capped at a configurable maximum number of examples to fit the budget. Run multiple seeds: 3 by default for small models.
- **Control run** *[my call; important]*. Also fine-tune on an equally sized **control dataset** of generic instruction data from a fixed public set, with a license the builder checks. Use the same number of seeds as the tuned runs, so the headline confidence interval includes the control's own seed variance. Any fine-tune on question/answer-shaped data can raise multiple-choice scores just by teaching the *format*. The report must separate:
  - **Raw gain:** tuned minus base.
  - **Data-specific gain:** tuned minus control. This is the headline number.
- **Evaluation suites**, scored base vs. control vs. tuned, each a fixed-size subset recorded in the run:
  - **Target:** medical QA benchmarks (MedQA/USMLE, MedMCQA, PubMedQA).
  - **Adjacent:** related science subjects (for example the MMLU anatomy, biology and chemistry subsets).
  - **Side effects:** general abilities (for example other MMLU subjects, ARC-Challenge and a GSM8K subset). The builder may add an instruction-following check.
- **Scoring:** log-likelihood multiple-choice scoring, which avoids parsing generated text. The builder decides whether to use lm-evaluation-harness or a small custom scorer.
- **Uncertainty:** every delta is reported in **percentage points with a 95% confidence interval**, combining bootstrap over questions with variance across seeds.
- **Baseline check:** before any training, confirm the base model scores clearly above chance on the target suite. If it doesn't, stop with a warning recommending a larger model or an easier suite. A gain measured from a random-guessing baseline is noise.

### 5.3 Part 3: Scaling curve (how much is enough?)

- Fine-tune on nested subsets of 10%, 25%, 50%, 75% and 100% of the data. Use a fixed shuffle so smaller subsets are contained in larger ones. The 100% point reuses part 2.
- Also fine-tune the **control data at each matching size**. That makes every point on the curve a data-specific gain (tuned minus control at the same size). It also gives a generic-data curve, which is one of the conversion anchor's ingredients (section 6). *[my call]*
- Curve points below 100% run one seed each. Their uncertainty is borrowed from the part 2 seed spread, and the report labels it as borrowed.
- Plot data-specific gain against the number of examples. Fit a saturating curve (for example a power law with a plateau) and report:
  - how many examples are needed to reach 80% of the plateau gain;
  - an extrapolated "what more data would add" estimate;
  - the fit's uncertainty. Five points is not much, so the report must say so.

### 5.4 Part 4: Which parts matter?

- **Contamination check:** n-gram overlap between the dataset and every evaluation item, using something like the 13-gram method from the GPT-3 paper. The builder may tune n. If matches are found:
  - flag the matching examples;
  - rerun the impact test with them removed;
  - report **raw gain vs. clean gain**.
- **Cluster contributions:**
  - Embed the examples with a small sentence-embedding model and group them into k clusters. The default is 4, configurable *[my call]*. Removing one cluster then takes out roughly a quarter of the data, which is enough to measure.
  - Label each cluster with top terms and sample items.
  - Run a **leave-one-cluster-out** fine-tune per cluster. Removing a cluster also means training on less data, so don't compare against the full run. Instead, compare against the part 3 scaling curve's predicted gain at the same reduced size:

    *contribution = curve-predicted gain at (N − cluster size) − measured gain without the cluster*

    This separates "this cluster mattered" from "there was less data". *[my call]*
  - If a contribution falls inside seed noise, the report says "not distinguishable from noise" instead of ranking it.
  - This ignores interactions between clusters, and the report says so.
  - Show each cluster's part 1 novelty mix next to its contribution.

### Report output

- A terminal summary.
- A human-readable report file (Markdown; static chart images such as the scaling curve are optional).
- The corpus record (section 6).

The report leads with the data-specific gain on the target suite, followed by side effects, the novelty mix, the scaling curve, contamination and clusters. Every number carries its model and context.

### Run profiles

- **`quick` profile:** part 1, plus part 2 with a single seed (tuned and control) and a small cap. Target: about an hour on the M5 for a 0.5–1.5B model.
- **`full` profile:** everything. Honest uncertainty is non-negotiable (section 4), so `full` may take more than one night locally. *[my call]*

The `full` run plan for a 0.5–1.5B target:

| Part | Fine-tunes |
|---|---|
| Impact test | 3 seeds tuned + 3 seeds control |
| Scaling curve | 4 tuned (10/25/50/75%) + 4 control at matching sizes |
| Cluster contributions | 4 |
| Contamination rerun | 1, only if contamination is found |

That's about 18–19 fine-tunes, each followed by an evaluation pass. Each fine-tune plus its evaluation is a separate checkpointed step. A run can therefore be stopped and **resumed** across nights, or individual steps can be moved to a rented GPU.

The builder measures real run times on the M5 early, sizes the evaluation subsets so evaluation doesn't dominate, and publishes an estimated time per profile.

## 6. Corpus record

One record per run, saved in an append-only local store. The builder picks the format (JSONL or SQLite). Each record contains:

- **Run:** ID, timestamp, engine version, library versions, hardware/device, seeds, all hyperparameters, run profile.
- **Dataset fingerprint:** content hash, number of examples, token-length stats, language, declared domain/type, declared date range, duplicate stats, cluster summaries.
- **Model:** ID, revision hash, parameter count, family, knowledge cutoff if known.
- **Results:**
  - part 1 statistics, including exposure reliability;
  - part 2 base/control/tuned scores per benchmark, with CIs and per-seed values;
  - part 3 curve points and fit;
  - part 4 contamination findings and cluster contributions.
- **Compute used:** device, wall-clock time, tokens trained, estimated training FLOPs.
- **Conversion anchor ingredients** *[my call]*. The anchor's exact definition is decided later, so v1 records ingredients for both candidate definitions:
  - **"More generic fine-tuning instead":** the control-data scaling curve from part 3.
  - **"A bigger model instead":** untuned evaluation scores of the larger sibling model(s) in the same family on the same suites. Also record their parameter counts and published pretraining token counts, from which training compute can be estimated. The part 1 reference model is already loaded, so this needs only evaluation passes, no training.

The record must be complete enough that:

- a third party can reproduce the run;
- a future predictor can learn (dataset fingerprint + part 1 signals + model/context) → (parts 2–4 outcomes);
- the conversion anchor can be calculated later from the ingredients above, whichever definition is chosen, without rerunning anything.

## 7. How it's built

**Decided by the owner:**

- Python. The owner writes Python and will work on the code.
- Hardware: MacBook Pro M5, 16 GB unified memory.
- Cloud budget: about **$50 total** for the first concept, for occasional rented-GPU runs.
- Output is an impact breakdown, with money deferred.
- An open, verifiable tool, CLI first.
- The four report parts and the corpus.
- The conversion anchor stays in the long-term design.

**My calls** (the owner reviewed these and didn't object):

- **Stack:** PyTorch, Hugging Face `transformers`, `peft` for LoRA and `datasets`. Runs on Apple's GPU (MPS) locally and on CUDA in the cloud from one codebase. Apple's MLX was rejected because it's Mac-only and would need a second codebase.
- **Models:** a current open model family with several sizes and a permissive license (for example Qwen 2.5/3 or Llama 3.x).
  - Local targets: about 0.5–3B.
  - Part 1 reference model: a larger sibling that fits in memory.
  - Validation: about 8B on a rented GPU with 4-bit LoRA (QLoRA).
- **First test case:** medical QA, with the benchmark suites in section 5.2.
- **Example CLI shape** (the builder may refine it):
  - `dvx scan DATA --model M` runs part 1 only.
  - `dvx run DATA --model M --domain medical-qa --profile quick|full` runs the full report.
  - `dvx resume RUN_ID` continues a stopped run.
  - `dvx report RUN_ID` re-renders a report.
  - `dvx corpus list|export` manages the corpus.

**Calls made after the design review**, not yet seen by the owner, are marked *[my call]* in sections 3, 5 and 6:

- the domain suite registry;
- the dataset size limit and sampling;
- the run plan with resumable runs;
- matched control curves;
- curve-based cluster attribution;
- the conversion anchor ingredients.

**Left to the builder:**

- package layout
- exact thresholds, hyperparameters and evaluation subset sizes
- the clustering method and the embedding model
- the storage format
- whether to use lm-evaluation-harness
- the report layout
- the test strategy, within section 8

## 8. How we know v1 works

**Known-answer tests.** The engine must give the expected answer on datasets where we already know the truth:

1. **Helpful data.** A clean sample from a medical QA training split (for example MedMCQA train) shows a positive data-specific gain on the target suite, with its CI above zero.
2. **Corrupted data.** The same sample with shuffled answers shows no gain or a negative one, and part 1 classifies much more of it as noise.
3. **Contaminated data.** The helpful sample with benchmark test items inserted is flagged by part 4, and the clean gain falls below the raw gain.
4. **Duplicated data.** A small sample repeated many times is flagged as heavy duplication, and its scaling curve is flat.
5. **Exposure calibration.** The part 1 exposure signal reports its AUC on the known-seen/unseen control set. Whatever the result, it is reported honestly.

**Transfer check** (within the $50 budget): run parts 1–2 on 2–3 of these datasets with an ~8B model on a rented GPU. Check whether the *direction and ranking* of results match the small models. Record the agreement in the corpus and the docs. This is the evidence for or against using small models to build the corpus.

## 9. Known risks

- **MPS speed and nondeterminism.** Fine-tuning on Apple GPUs is slower than on CUDA, and not bit-for-bit deterministic. Multiple seeds and reported variance handle this. Run times must be measured early.
- **Small models near chance on hard medical exams** (MedQA especially). The baseline check guards against this; it may force a 1.5–3B target or a heavier reliance on PubMedQA and MedMCQA.
- **The membership-inference signal may be unreliable.** That's handled by calibration and "inconclusive" reporting.
- **Small-to-large transfer may not hold.** That's what the transfer check is for.
- **Benchmark and dataset licenses** must be checked before anything is redistributed.

## Glossary

- **Fine-tune / LoRA:** briefly training a model on new data. LoRA trains small add-on weights instead of the whole model, which is cheap enough for a laptop.
- **Loss / surprise:** how unexpected a piece of text is to the model. Lower means it already "knows" it.
- **Membership inference:** estimating whether specific text was in a model's training data.
- **Contamination:** evaluation (test) questions appearing in training data, which makes gains look bigger than they are.
- **Confidence interval (CI):** the range the true value likely falls in. A gain whose CI includes zero isn't established.
- **Corpus:** the growing collection of saved run records, the "data about data value".
