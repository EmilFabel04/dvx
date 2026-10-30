# dvx

**Measure what a dataset actually does to a model, before you buy it or train on it.**

Today datasets are sold by describing what they *are*: size, type, license, a sample. Nobody says what they will *do* to a model. dvx is an open, reproducible engine that takes **one dataset and one target model** and produces a **Data Impact Report**: a breakdown of what that data changes in that model, in that context.

It is a tool you can check, not a score you have to trust. Anyone with the same inputs can rerun it and get the same result.

> **Status: pre-alpha, design stage.** Nothing is runnable yet. The v1 design is in [`docs/superpowers/specs/2026-10-30-data-impact-engine-design.md`](docs/superpowers/specs/2026-10-30-data-impact-engine-design.md).

## What the report answers

1. **What doesn't the model already know?** A fast scan with no training. It measures how surprised the model is by each example and splits the data into *already known*, *learnable* and *likely noise*. It also flags duplicates and gives a calibrated estimate of whether the model has seen this data before.
2. **What does it change?** A short LoRA fine-tune, scored before and after on target, adjacent and general benchmarks. The result is reported against a generic-data control and with confidence intervals. Side effects are shown as prominently as gains.
3. **How much is enough?** A scaling curve of gain against data size, including the point where more data stops paying off.
4. **Which parts matter?** Benchmark-contamination detection (raw gain vs. clean gain) and cluster-level contributions.

Every run also writes a structured record to a local **corpus** of "data about data value". Over time that corpus is meant to train a predictor that estimates impact without a training run.

## Principles

- **No single "intelligence score."** Every number carries its model, its context and its uncertainty.
- **Reproducible.** Inputs, model revisions, seeds and versions are pinned and recorded.
- **Honest about weak signals.** A signal that can't be validated is reported as inconclusive.

## v1 scope

- Text datasets in prompt/response or chat format.
- Medical QA as the first evaluation domain.
- Open models of about 0.5–3B parameters on a laptop (Apple Silicon via PyTorch MPS), with validation on about 8B models via CUDA.
- A CLI and a Python library. Dollar values are explicitly out of scope for v1.

## License

[Apache-2.0](LICENSE)
