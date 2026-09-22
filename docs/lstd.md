# Frozen-policy LSTD with a one-layer critic

Fit `V(h) = wᵀ normalized(h) + b` to the cached base-Qwen trajectories.
This is the **raw linear value head**, not the sigmoid/BCE probe. LSTD fits new
weights; it does not reuse the supervised head's weights or change the actor.
No generation, model loading, verifier installation or GPU is required.

## Run on Duke

After publishing the code changes to GitHub, on the cluster:

```bash
cd /usr/xtmp/ms785/rl_ms_project
git pull --ff-only origin main
source scripts/cluster_env.sh
sbatch scripts/submit_lstd.sh outputs/critics-12654337
```

The script requests 8 CPUs, 64 GB RAM and 12 hours on `compsci`. Runtime depends
on reading the full feature cache and dense matrix accumulation, not just the
small final solve. The 32-second ridge timing from the earlier prefix-sampled
experiment is not a runtime estimate for this all-transition experiment.

```bash
squeue -j JOB_ID
tail -F slurm-lstd-JOB_ID.out
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed
cat outputs/lstd-JOB_ID/report.txt
```

Expect progress every 25 training questions, then a comparison of LSTD and ridge.
A lower test Brier is better; a negative paired LSTD-minus-ridge difference favors
LSTD. Inspect calibration, out-of-range predictions, conditioning and costs in
`summary.json`. There is no automatic promotion to PPO.

If interrupted, resume with the **old output directory**:

```bash
sbatch scripts/submit_lstd.sh outputs/critics-12654337 --out outputs/lstd-OLD_JOB_ID
```

Atomic `statistics.pt` checkpoints preserve accumulated matrices every 25
questions. Resume checks source hashes, settings, runtime and code; it refuses
changed processed shards. The source experiment is never edited. A new Slurm
job has a new log name even when its output directory is reused.

## Equations and relationship to the paper

The reference is [Lagoudakis & Parr (2003), section 5.2](https://www.jmlr.org/papers/v4/lagoudakis03a.html).
The paper also develops LSTDQ/LSPI for action values and policy improvement.
Here we use its **fixed-policy state-value prediction** formulation, not the
LSPI control loop or a vocabulary-sized action-value function.

For each pre-token feature vector `phi`, append a constant intercept. With
`gamma = 1` and `lambda = 0`, accumulate in float64:

```text
A = mean(phi_t (phi_t - phi_next)^T)
b = mean(phi_t reward_t)
(A + alpha D) w = b
```

`D` is identity except for the unpenalized intercept. Regularization shifts the
empirical TD fixed point. We solve the nonsymmetric system with
`torch.linalg.solve`; we do not invert it, minimize squared TD residuals, or
silently switch to a pseudoinverse. Record equation residuals and the selected
system's condition number. Finite, numerically accurate solutions are not proof
of accurate values or a well-conditioned problem.

Reference packages exist: [LSPI-Python](https://pythonhosted.org/lspi-python/)
and [MushroomRL's LSPI source](https://mushroomrl.readthedocs.io/en/latest/_modules/mushroom_rl/algorithms/value/batch_td/lspi.html).
They focus on action-value/control APIs. Our small PyTorch implementation follows
the same matrix construction, adapted to saved token states and a fixed actor;
it is not vendored code or a claim to use either package directly.

## Frozen protocol

- Reuse the original train/validation/test question split and train-only feature
  normalization. All 1,536 hidden dimensions plus intercept: 1,537 fitted parameters.
- Use every retained training transition once. Longer answers therefore contribute
  more weight. This differs from the previous eight-prefix supervised fit.
- Rewards are zero until the last action, which receives the stored binary outcome.
  EOS **and the 2,048-token cap** end the task. Terminal next features are entirely
  zero, including the intercept, after normalization. Do not connect episodes.
- Preserve source exclusions: only nonempty `correct`/`incorrect` responses have
  cached features. Parsing failures and verifier errors remain excluded; do not
  quietly change their rewards. Results are conditional on this selected dataset.
  Audit exclusions before interpreting the critic as the value of the full policy.
- Fit a new ridge comparator on exactly the same pre-action states and weighting,
  with final outcome as its target. Both methods tune the same positive alpha grid
  using validation Brier. Adding alpha has different mathematical effects in the
  two objectives; this is a method comparison, not identical regularized losses.
- Select both methods before loading test tensors. Evaluate with the existing
  random-prefix protocol, not all-transition weighting. The test set has already
  been inspected in the supervised study, so this is exploratory comparison.
- Report a paired question-bootstrap interval, raw Brier, accuracy, AUROC,
  calibration, out-of-range rate, solver diagnostics and shared accumulation cost.
- Save normalized-coordinate weights/bias and normalization in `lstd.pt` and
  `ridge.pt`. No sigmoid or clipping is applied to reported predictions.

The normalization comes from sampled training prefixes, but both solvers use it
identically. A hidden vector need not be a sufficient Markov state. The terminal
cap's remaining budget is not an explicit feature. These are approximation limits.
If return regression works but LSTD(0) does not, inspect conditioning and then
consider LSTD(lambda) for long sparse-reward sequences, rather than concluding
that token-level critics cannot work. Online PPO is a later experiment using
fresh data as the actor changes; this run makes no online learning claim.
