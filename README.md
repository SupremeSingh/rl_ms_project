# SLICE: a compact critic for PPO

**Can an LLM learn with PPO without training a second transformer as its critic?**
This project demonstrates that approach on numeric math completion. The math
experiment is complete; this branch contains the reusable SLICE package and the
final six-method Numina pipeline.

## Start with one answer

A model answering “9 avocados, 3 per serving” chooses one token at a time. At
“Divide 9 by…”, the critic estimates the chance that continuing will earn a correct
final answer. Math-Verify supplies a terminal reward of 1 or 0. GAE combines
rewards and token-level values; PPO adjusts the actor's token probabilities.

SLICE supplies those values cheaply:

**Question + partial answer → frozen Qwen → selected hidden layers → sparse 768 features → linear critic.**

We read blocks **28, 27, 25, 21, 13** from Qwen2.5-Math-1.5B, normalize and
concatenate their 1,536-dimensional vectors, and train a fixed ReLU compressor
from 7,680 to 768 dimensions. Sparse means an activation penalty during training;
the implementation still uses dense operations.

During PPO, the encoder and compressor stay frozen. Two persistent question-fold
systems accumulate Ridge or LSTD statistics. Each question receives values from
the opposite fold, keeping its own rewards out of its head fit. We solve for
linear weights, compute advantages and update only the actor. There is no buffer,
encoder refresh or critic-transformer backpropagation in the SLICE arms.

[System architecture](docs/figures/final-architecture.svg) ·
[Package API](packages/slice-critic/API.md) · [Setup](SETUP.MD) ·
[Reproduction](docs/REPRODUCING.md)

## Final experiment

Five actor seeds, six methods, **120 updates × 64 answers**, 2,000 test questions,
fixed greedy evaluation and a 2,048-token response cap. Gamma is 1; GAE lambda is
0.95; online LSTD lambda is 0.9; cumulative regularization is 0.01. Each run uses
two A5000 GPUs. Offline feature learning and PPO validation calibration are
additional costs.

| Method | Accuracy | Training GPU hours, mean |
|---|---:|---:|
| Sparse-768 Ridge PPO | **74.60%** | **10.35** |
| Conventional PPO | 73.52% | 12.51 |
| GRPO | 72.75% | 9.38 |
| Two-thirds-layer Ridge PPO | 72.54% | 10.15 |
| Two-thirds-layer LSTD(0.9) PPO | 72.40% | 9.61 |
| Sparse-768 LSTD(0.9) PPO | 70.86% | 10.33 |

Sparse Ridge gained **2.06 percentage points** over two-thirds Ridge; its adjusted
interval was **[0.59, 3.53] pp**. Against GRPO the gain was **1.85 pp**, interval
**[0.08, 3.50] pp**. Against conventional PPO the gain was **1.08 pp**, interval
**[-1.22, 3.56] pp**: superiority over PPO is unresolved. Measured online training
cost was about 17% below conventional PPO, excluding offline feature learning.

These are approximate seed/question bootstrap intervals with Bonferroni nominal
coverage for eight primary comparisons. Five actor seeds share one selected
compressor. Eligibility uses numeric-template separation, not semantic or
pretraining decontamination. Verifier labels need audit. One conventional PPO seed
failed the training-target-MSE improvement gate; its other functional checks
passed, and it remains in the comparison. No seeds are discarded.

[Recorded results and per-seed data](docs/results/numina-six-method.json).
Compression worked well here; sparsity itself has not been shown to speed up the
implementation. Ridge is our practical default. LSTD remains a supported
alternative, not an established winner for long horizons.

## What lives where

| Location | Responsibility |
|---|---|
| `packages/slice-critic/` | Frozen features, compressor training, Ridge/LSTD, verification, artifacts and scoring; independent of VERL. |
| `src/math_rl/` | Numina preparation, PPO/GRPO, GAE integration, validation and reporting. |
| `scripts/` | Thin commands, environment setup, Slurm submission and smoke checks. |
| `configs/` | The two active VERL configuration files. |
| `tests/` | Tests for the supported production workflow; package tests live with the package. |

The package remains **internal**, installed from source or a checked wheel; there
is no hosted API, registry or PyPI publication. Completed v1 artifacts remain
supported. Both standalone training/scoring and real PPO integration passed on
Duke before this cleanup. The refactored branch requires its own cluster smoke;
a branch named `prod` is not a promise of fault-free behavior.

## Reproduce or start fresh

The [reproduction guide](docs/REPRODUCING.md) provides two routes: train fresh
actors from the final locked data/filter/profile, or build a new SLICE artifact
and family-disjoint Numina study from pinned data. Retraining the representation
is a new experiment; it is not a promise to recover identical percentages.
Older exploratory code is available in Git history, not in the active workflow.
Saved outputs, model weights and environments are never deleted by setup.

The first stage established a working, lower-cost linear-critic path for PPO on
this task. Value-guided Lean search is a future application, not an implemented
or validated capability of this repository.
