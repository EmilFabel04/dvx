# M5 quick-profile run: timings

Machine: Apple M5, 16 GB (`env.device_name`, `env.ram_bytes`). Target model:
Qwen/Qwen3-1.7B-Base (1,720,574,976 params), revision `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`.
Reference: Qwen/Qwen3-4B-Base, revision `906bfd4b4dc7f14ee4320094d8b41684abff8539`.
Profile: quick (1 seed, 250 optimizer steps per arm). Data: 2,000 MedMCQA train examples.
Eval protocol: suite v2, plain-text 5-shot letter scoring, eval max length 3,072, eval prompts
truncated: 0. Engine commit: `a7e8d36` (clean, `env.dvx_dirty` is false). Run: `20261101-233734-8c7ff65f`.
Training examples truncated to 512 tokens: 71 in the tuned arm, 115 in the control arm.

Total wall time 8,011 s (2 h 13 min 31 s): the sum of the step walls in `record.json`
(`compute.steps`), rounded once. Process wall time from the log's start timestamp (01:37:33) to
the watchdog's observation that the process was gone (03:51:48) is about 8,055 s (2 h 14 min),
approximate to within 35 s (30 s polling plus 5 s grace). There was no `/usr/bin/time` for this run.
All Hub downloads and the calibration set were already cached.

## Wall time per step (`record.json`, `compute.steps`)

| Step | Wall time | Seconds |
|---|---|---|
| dataset | under 1 ms | 0.0 |
| duplicates | 0.9 s | 0.9 |
| contamination | 0.4 s | 0.4 |
| scan_target | 6 min 29 s | 389.1 |
| scan_reference | 7 min 1 s | 421.5 |
| eval_base | 16 min 13 s | 972.7 |
| tuned_s0 | 51 min 33 s | 3,092.5 |
| control_s0 | 52 min 14 s | 3,133.8 |

Steps ran in the order shown.

## Training detail (`steps/*_s0.json`, `train` and `_wall_seconds`)

| Arm | Step wall (s) | Train-only (s) | Eval remainder (s) | Remainder share of step | Tokens trained | Tokens/s (train-only) |
|---|---|---|---|---|---|---|
| tuned | 3,092.5 | 1,661.7 | 1,430.9 | 46.3% | 348,888 | 210.0 |
| control | 3,133.8 | 1,703.9 | 1,429.9 | 45.6% | 306,732 | 180.0 |

Eval remainder = step wall seconds minus train-only seconds.

Each arm's eval remainder (about 1,430 s) is longer than eval_base (972.7 s) for the same 12
benchmarks. A likely cause is the unmerged LoRA adapter's extra matmuls. This is a hypothesis,
not measured.

## Memory

What was measured, and how:

- A watchdog sampled the Python process footprint every 30 s. The highest sample was
  10,240 MB, taken at 01:45:20, about a minute into scan_reference (the 4B reference model).
  A spike shorter than 30 s could be missed.
- Two controller spot checks, not logged to a file: during tuned_s0 training, footprint
  6,714 MB and GPU device utilization 99%; during control_s0, about 40 min in, footprint
  8,796 MB and GPU 88%.
- An earlier probe (final-fix report, G8) with Qwen3-4B-Base scoring 32 examples: peak
  9.85 GiB MPS driver memory and 10.29 GiB process footprint.
- Nothing reached the 11 GB alert.

## Results

| Item | Value |
|---|---|
| Baseline margin over chance (target) | +15.6 pp (need 5.0 pp), passed |
| Chance = max(1/k, majority share) | medqa 29.2%, medmcqa 35.6%, pubmedqa 56.2% |
| Headline data-specific target delta | +0.9 pp (95% CI -0.6 to +2.2), not distinguishable from zero |
| Data-specific adjacent | -1.3 pp (95% CI -3.5 to +0.9) |
| Data-specific side_effect | -0.1 pp (95% CI -1.4 to +1.3) |
| Raw (tuned - base): target / adjacent / side_effect | +1.3 / -2.0 / +0.2 pp |
| Format effect (control - base): target / adjacent / side_effect | +0.5 / -0.7 / +0.2 pp |
| Contamination | 0 flagged examples |
| Novelty mix | known 10.1% (201), learnable 47.2% (945), noise 42.7% (854) |
| Novelty reliability | reference below target on 96.3% of examples (gate 50%); paired discrimination 1.00 (gate 0.60) |
| Prior exposure | inconclusive, calibration AUC 0.52 (gate 0.6) |
| Duplicates | exact 0, near 0 |

## Projected `full`-profile duration (1.7B-Base, this machine)

Formula: 19 x mean(tuned_s0, control_s0) + scan_target + scan_reference + eval_base

- mean(3,092.523, 3,133.804) = 3,113.164 s
- 19 x 3,113.1637 (unrounded mean) = 59,150.1 s
- plus 389.1 + 421.5 + 972.7 = 1,783.3 s
- Total = 60,933.5 s (from unrounded values) = **about 16.9 hours**

Each fine-tune step already includes its own eval. This is an upper bound if M2's
scaling-curve fine-tunes use fewer examples than the 2,000 used here.

The previous doc extrapolated about 20 hours for 1.7B from the 0.6B run. This measurement is
about 15% lower (16.9 / 20 = 0.85).

## What this run says about the tool

- **The 5-shot eval removed the format inflation.** The format effect (control - base) is
  +0.5 / -0.7 / +0.2 pp for target / adjacent / side_effect, all with CIs that include zero. The
  first run's raw gains were +6.0 / +18.0 / +16.5 pp, mostly format learning. The control arm
  is still worth keeping as a check, but it no longer dominates.
- **A quick fine-tune on these 2,000 MedMCQA examples does not measurably move 1.7B-Base on
  medical QA.** Data-specific target +0.9 pp, CI -0.6 to +2.2, which includes zero. In-domain
  MedMCQA accuracy: base 51.4, tuned 51.0 (control 51.0). That is a result about this dataset
  and profile, not a tool failure, and it is one seed.
- **The novelty reliability gate passes, but the "noise" bucket is confounded with response
  length.**
  - (a) On all 200 mismatched pairs, the mismatched reference loss exceeds the true pair's.
    The gap is at least 0.069 nats per token, median 0.491.
  - (b) The noise threshold (1.902 nats per token) is the 10th percentile of mismatched
    reference losses, compared with each example's *absolute* reference loss. Noise =
    reference loss >= 1.902 reproduces the record's 854 examples (42.7%).
  - (c) Noise-labelled responses have a median length of 457 characters, against 234 for the
    others (854 and 1,146 examples).
  - (d) A plainly clean item is labelled noise: line 8, "Name of mumps vaccine is ... Jeryl
    Lynn" (reference loss 2.337, response 255 characters).

  Conclusion: the M2 novelty redesign should use the per-example paired gap rather than an
  absolute threshold. A paired discrimination of 1.00 also shows that mismatched pairs are an
  easy test.
- **Exposure is inconclusive again** (AUC 0.52) with Wikipedia calibration. M2 needs in-domain
  calibration or another membership signal.
- **Evaluation is now a large share of each fine-tune step:** 46.3% of the tuned step and
  45.6% of the control step is eval remainder.
- The record's `calibration_set.fetched_on` is null because the cached set predates that field.
  Its SHA-256 (`7f1fe785...c8e85`) is recorded.

## Previous run (superseded)

Run `20261101-201145-8c7ff65f`, commit `5ecee1b`: post-trained Qwen3-0.6B, zero-shot
chat-template eval, total 3,053 s. Its novelty mix (80.5% noise) was an artifact of post-trained
checkpoints, and its single contamination flag was a stock-phrase false positive. Its numbers
are not comparable with this run.
