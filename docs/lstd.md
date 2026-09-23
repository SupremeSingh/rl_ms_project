# Frozen-policy LSTD(lambda) with a one-layer critic

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

Expect progress every 25 training questions, then all five lambda results and ridge.
The default sweep is `--lambdas 0 0.9 0.99 0.999 1`. This requires more accumulation
work than the earlier lambda=0 run (about 32 minutes of shared accumulation).
Do not reuse the earlier `outputs/lstd-12686052` directory: the protocol changed.
A lower test Brier is better; a negative paired selected-LSTD-minus-ridge difference
favors LSTD. Select lambda on validation Brier, not the displayed test scores.
Inspect calibration, out-of-range predictions, conditioning and costs in
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

The trace extension follows [Boyan (2002)](https://www.cs.cmu.edu/afs/cs/user/jab/web/cv/pubs/boyan.lstdl-mlj.pdf).
For each pre-token feature vector `phi`, append a constant intercept. With
`gamma = 1`, accumulate in float64:

```text
z_t = lambda * z_previous + phi_t
A = mean(z_t (phi_t - phi_next)^T)
b = mean(z_t reward_t)
(A + alpha D) w = b
```

Reset `z_previous` to zero at every new answer; retain it across computation
chunks within that answer. Lambda=0 is the original LSTD(0) experiment. Lambda=1
recovers return regression over complete capped episodes. A parallel prefix scan
computes traces without an expensive Python loop over every token.

The run checks lambda=1 against ridge's accumulated matrix and right-hand side
(relative error <= 1e-9), and validation predictions at ridge's chosen alpha
(max absolute difference <= 1e-6). It fails explicitly if these checks do not hold.
This is an implementation check, not independent evidence that lambda=1 beats ridge.

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
  with final outcome as its target. All methods tune the same alpha grid, including zero,
  using validation Brier. Singular or inaccurate solves are recorded and rejected.
  Then select the LSTD lambda on validation Brier before loading test tensors. Adding alpha has different mathematical effects in
  the
  two objectives; this is a method comparison, not identical regularized losses.
- Evaluate all candidates with the existing
  random-prefix protocol, not all-transition weighting. The test set has already
  been inspected in the supervised study, so this is exploratory comparison.
- Report a paired question-bootstrap interval for the validation-selected LSTD
  against ridge, raw Brier, accuracy, AUROC, prediction spread, calibration,
  out-of-range rate, solver diagnostics and shared accumulation cost. Per-lambda
  test results are exploratory; do not pick a different winner using them.
- Save normalized-coordinate weights/bias and normalization in per-lambda
  checkpoints (`lstd.pt` for zero, `lstd-0.9.pt`, etc.), `selected-lstd.pt`, and
  `ridge.pt`. No sigmoid or clipping is applied to reported predictions.

The normalization comes from sampled training prefixes, but both solvers use it
identically. A hidden vector need not be a sufficient Markov state. The terminal
cap's remaining budget is not an explicit feature. These are approximation limits.
If only lambda=1 performs well, that favors supervised return fitting in this
setting; it does not establish a benefit from bootstrapping. Intermediate lambdas
may or may not improve the tradeoff. Online PPO is a later experiment using
fresh data as the actor changes; this run makes no online learning claim.

## Length diagnostics and seed replication

Run `outputs/lstd-12687136` selected lambda=0.999, alpha=0.001. Its test Brier was
0.19751 versus ridge 0.19845. These checks keep both fitted critics fixed.

```bash
sbatch scripts/submit_lstd_analysis.sh outputs/lstd-12687136
sbatch scripts/submit_lstd_validation.sh outputs/lstd-12687136
```

**Existing data (2 CPUs, 4 GB, 30 minutes):** compare saved predictions in fixed
1–128, 129–512 and 513+ token bins, by total answer length, prefix position and
remaining actions T-L. Position zero is separate. Also report position crossed
with remaining length, and capped/uncapped answers. Length is analysis metadata
only; it never enters the critic. Reconstruct sampled-prefix ordering and check
it against saved question IDs, positions and labels before comparing predictions.

Report pooled-prefix Brier for each head, outcome rates, answer/question counts,
and equal-question paired differences with 2,000 question-bootstrap samples.
Groups with fewer than 20 questions are flagged sparse. Intervals are descriptive,
not multiplicity-adjusted. Differences in difficulty, question mix and survival
mean these are not causal effects of reasoning length.

**New generation seed (one A5000, 64 GB, 24 hours):** generate 16 new answers on
each of the same 500 test questions, 8,000 total, using seed 314159 plus local
question index. Preserve prompts, sampling settings, model bytes, runtime versions,
verifier and exclusions. Evaluate the previously selected LSTD and ridge heads
without refitting or retuning. Checkpoints and normalization are locked by hashes.

Generation, extraction and evaluation run in separate processes. Per-question
outputs are resumable. New answers and features live in a separate directory;
original data and fitted critics are unchanged. The fixed prefix-sampling rule
is reward-blind; realized length is used only for evaluation diagnostics.
This tests sensitivity to sampled continuations on the SAME questions, not fresh
questions or independent training fits. Novel-question confirmation is future work.

```bash
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed
cat outputs/lstd-analysis-ANALYSIS_JOB_ID/report.txt
cat outputs/lstd-validation-VALIDATION_JOB_ID/status.json
cat outputs/lstd-validation-VALIDATION_JOB_ID/report.txt
cat outputs/lstd-validation-VALIDATION_JOB_ID/length-report.txt
```

Logs: `slurm-lstd-analysis-JOB_ID.out`, `slurm-lstd-validation-JOB_ID.out`, and
`generate.log`, `extract.log`, `evaluate.log` inside the validation output.
Resume an interrupted validation run with its original output directory:

```bash
sbatch scripts/submit_lstd_validation.sh outputs/lstd-12687136 \
  --out outputs/lstd-validation-OLD_JOB_ID
```

A consistent advantage supports stability but does not prove superiority or
cheaper PPO. Compare exclusion rates too: missing answers can change the evaluated
population. Preserve all original and new results.
