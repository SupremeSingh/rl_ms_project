# Conventional PPO baseline protocol

The previous MATH pilots established that the machinery runs, not that conventional
PPO was well tuned. We now calibrate that baseline before repeating the comparison.
The buffer and explicit planning are disabled.

## What remains standard

The pinned VERL implementation supplies causal pre-token values, GAE (gamma 1,
lambda .95), clipped policy loss, clipped value loss and AdamW. The critic learns
from detached **raw GAE returns**, computed before actor-advantage whitening.
The full critic backbone and scalar head are trainable; its optimizer persists.
All methods start from Qwen2.5-Math-1.5B, use the same numeric MATH splits,
Math-Verify, ordinary completion, 512/2048 token limits and two GPUs. Actor LR is
1e-6, two actor epochs, clip .2. No KL penalty is used in any method: this is a
controlled verifiable-reward comparison, not a full RLHF recipe replication.

[VERL's PPO reference](https://verl.readthedocs.io/en/latest/algo/ppo.html) describes
the actor–critic, GAE and clipping mechanisms, and distinguishes memory-oriented
microbatch settings from algorithmic batch sizes. Its published example uses
larger batches than our resource-limited experiment. We do not claim those results.
The [RLHF implementation study](https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo)
documents zero value-head initialization and the importance of implementation
details. It concerns different tasks and is guidance, not evidence for our MATH results.

## Changes to investigate

The previous critic used `seq-mean-token-sum-norm`, dividing summed token loss by
padded response width. We use `seq-mean-token-mean`: mean over valid tokens in each
answer, then mean over answers. With fixed microbatch one and gradient accumulation,
this weighting is independent of padding width. It differs from a global token
mean on unequal-length answers. This is an explicit weighting change, not proof
that the previous weighting caused failure. See the pinned
[loss definitions](https://github.com/verl-project/verl/blob/8d9e350ea58c7ad4b50dd14d9dcb50577242c55f/verl/trainer/ppo/core_algos.py).

The scalar head starts at zero, keeping the pretrained backbone unchanged.
Before/after-update diagnostics use the same raw return targets and mask. They
check fitting, not generalization. Extra evaluation forwards are included in
training cost and separately timed; reports also estimate cost excluding those
forwards, so instrumentation is not mistaken for an algorithmic saving. Value
clipping remains .5.

## Selection and locked comparison

Three candidates on seed 17, 30 updates each: LR 1e-5/2 critic epochs,
5e-5/2 epochs, 1e-5/4 epochs. All must have complete finite diagnostics,
zero initial predictions, nonzero learning, and lower aggregate post-update
training MSE. Final validation accuracy must not regress from initial validation.
Among eligible candidates, highest final validation accuracy wins; ties follow
the declared candidate order. If none qualifies, stop. No test scoring occurs
in calibration, and no calibration weights enter final runs.

Lock the winner and run all four methods from base weights for 60 updates × 64
answers on each of seeds 42, 43, 44. Ridge/LSTD use current actor features and
question cross-fitting on the current batch. LSTD lambda .99 and regularization
.01 remain fixed. Report the extra PPO tuning cost; only PPO receives that tuning
budget. Rotate method execution order across seeds. Report per-seed results,
mean/sample SD, paired answer changes and descriptive seed/question bootstrap
intervals. Primary comparison: LSTD-PPO versus calibrated conventional PPO.
Other contrasts are secondary; no automatic superiority claim or multiplicity-
adjusted hypothesis test is made.

The test split has been inspected repeatedly. Three seeds, capped generations,
limited batches, incomplete reward audit and this small hyperparameter grid
still limit conclusions. A poor result after calibration is reportable; it is
not permission to retune against test answers. Untouched evaluation remains a
separate requirement for a confirmatory study.
