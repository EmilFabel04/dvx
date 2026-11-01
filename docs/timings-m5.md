# First M5 quick-profile run: timings

Machine: Apple M5, 16 GB. Model: Qwen/Qwen3-0.6B (596,049,920 params). Profile: quick (1 seed,
250 optimizer steps per arm). Data: 2,000 MedMCQA train examples. Engine commit: `5ecee1b`.
Run: `20261101-201145-8c7ff65f`. Total wall time 3,053 s (50 min 53 s, `/usr/bin/time`).
Hub downloads were already cached for this run.
Examples truncated to 512 tokens: 79 in the tuned arm, 120 in the control arm.

## Wall time per step (`record.json`, `compute.steps`)

| Step | Wall time | Seconds |
|---|---|---|
| dataset | under 1 ms | 0.0 |
| duplicates | 0.9 s | 0.9 |
| contamination | 0.4 s | 0.4 |
| eval_base | 1 min 38 s | 97.8 |
| scan_target | 3 min 40 s | 220.0 |
| scan_reference | 3 min 26 s | 206.4 |
| tuned_s0 | 20 min 37 s | 1,236.7 |
| control_s0 | 21 min 10 s | 1,270.1 |

## Training detail (`results.impact.train`)

| Arm | Step wall (s) | Train-only (s) | Eval remainder (s) | Tokens trained | Tokens/s (train-only) |
|---|---|---|---|---|---|
| tuned | 1,236.7 | 1,090.5 | 146.1 | 372,373 | 341.5 |
| control | 1,270.1 | 1,115.1 | 155.0 | 329,084 | 295.1 |

Eval remainder = step wall seconds minus train-only seconds.

One-time costs, not in the table above (cached afterwards): downloading Qwen3-1.7B took
6 min 22 s; fetching the Wikipedia calibration texts took about 23 min (cached in
`.dvx/calibration`).

## Memory

Peak memory was not observed with Activity Monitor. `/usr/bin/time -l` reports uv's own
footprint, not Python's, so it is not usable here. What we have:

- The controller sampled the Python process footprint every 30 s over the run. It never
  reached 10 GB.
- Bounded in-step diagnostics (PR #16): 3.4 GiB MPS driver memory and 4.1 GiB process
  footprint during training; 3.3 GiB peak driver memory during scoring.

## Results

| Item | Value |
|---|---|
| Baseline margin over chance (target) | +13.0 pp (need 5.0 pp), passed; no 1.7B rerun needed |
| Headline data-specific target delta | +0.7 pp (95% CI -1.5 to +2.6), not distinguishable from zero |
| Contamination | 1 flagged example (medqa: 1) |

## Projected `full`-profile duration (0.6B, this machine)

Formula: 19 x mean(tuned_s0, control_s0) + scan_target + scan_reference + eval_base

- mean(1,236.7, 1,270.1) = 1,253.4 s
- 19 x 1,253.397 (unrounded mean) = 23,814.5 s
- plus 220.0 + 206.4 + 97.8 = 524.2 s
- Total = 24,338.7 s = **about 6.8 hours**

Each fine-tune step already includes its own eval. This is an upper bound if M2's
scaling-curve fine-tunes use fewer examples than the 2,000 used here.

Rough 1.7B estimate: about 3x by parameter count, so about 20 hours. This is an
extrapolation, not a measurement.

## What this run says about the tool

- The raw gains are large in both arms: target +6.0, adjacent +18.0, side_effect +16.5 pp.
  The target and side_effect data-specific deltas are near zero. The control arm gains
  almost as much on target (+5.3 vs +6.0 pp), so most of the raw gain likely comes from
  learning the answer format. The control arm is essential.
- The data-specific adjacent effect is -3.8 pp (95% CI -7.5 to -0.4), which excludes zero.
  That is a measured harm against generic instruction data.
- Novelty labelled 80.5% of a clean, real dataset as "noise" and only 0.1% as "learnable".
  That is suspect; the classifier's thresholds need validation in M2.
- Exposure was inconclusive: calibration AUC 0.50 on 0.6B.
