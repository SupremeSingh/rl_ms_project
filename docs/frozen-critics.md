# Frozen critic experiment

Predict whether a partial answer will eventually be correct, using a small head
on frozen `Qwen/Qwen2.5-Math-1.5B` features. Math-Verify supplies the binary target.
Qwen does not train. This is supervised return prediction; LSTD and PPO come later.

## Data and prefixes

| Run | Train / validation / test questions | Answers per question | Total answers | Prefix examples |
|---|---|---:|---:|---:|
| Default | 128 / 32 / 32 | 8 | 1,536 | 12,288 |
| `--full` | 4,000 / 500 / 500 | 16 | 80,000 | 640,000 |

Prefix counts are upper bounds: responses without a reliable verifier label are
saved but excluded from feature extraction, fitting and test metrics. Wrong
answers successfully verified as incorrect remain in the dataset with target 0.

Counts above are planned totals before exclusions. Questions are split before generation. Previous audits, evaluation sets and
`outputs/critics-*/questions.json` are excluded, including the earlier critic
experiment. Keep those manifests in place. The official GSM8K test set is unused.
If too few eligible questions remain, selection fails rather than reusing them.

For each response of realized length `T`, sample eight states:

- One at `L=0`: the question, before any answer token.
- Seven independent uniform draws, with replacement, from `L=1,...,T-1`.
- If `T=1`, use eight copies of `L=0`.

`L` is the number of answer tokens already seen. The vector is the final
normalized hidden state at the end of the question plus those `L` tokens,
before choosing the next token. The sampler uses seed 1729 plus question and
response indices, independently of reward. All heads see the same saved samples.
Duplicates are intentional; every retained response has equal total weight.

This rule weights trajectories equally and depends on their realized length.
It changes the sampling distribution relative to the old fixed grid and to
online PPO states. Features remain causal, but these results are a supervised
diagnostic, not proof of unbiased on-policy value estimates. For later LSTD,
all consecutive states and token IDs are also retained.

## Fitting

- **One-layer MLP (`linear_logit`)**: linear output plus sigmoid, binary cross-entropy.
- **Two-layer MLP (`mlp2`)**: 256-wide GELU hidden layer, binary cross-entropy.
- **Ten-layer residual network (`resnet10`)**: 256-wide, binary cross-entropy.
- **Raw linear value (`linear_value`)**: no sigmoid, squared error.
- **Direct ridge regression**: an additional raw linear fit to returns, using
  float64 least squares with an unpenalized intercept. This is not LSTD.

All gradient heads start at the training success rate. They use three seeds,
learning rates `0.01, 0.001, 0.0001`, and up to **200 epochs per trial**. After at
least 30 epochs, stop after 30 consecutive epochs without a validation Brier
improvement greater than `1e-6`. Keep the best validation checkpoint, including
the initial constant predictor. A best checkpoint at epoch 200 triggers a warning;
the budget alone does not guarantee convergence.

Ridge selects its penalty on validation data from
`0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10`, minimizing mean squared training error
plus `alpha * ||weights||²`. It is a direct-solver diagnostic, not an exact match
to AdamW's regularization or fitting time.

Normalization, checkpoints and hyperparameters use training/validation only.
Test reports include accuracy, Brier, calibration, AUROC, matching constant
baselines by prefix, and uncertainty clustered by question. Fit curves record
training loss, validation error and time. The old and new aggregate scores have
different questions and prefix distributions, so are not a controlled comparison.

## Run on Duke

First push the updated code from your laptop. Then, on the cluster login node:

```bash
cd /usr/xtmp/ms785/rl_ms_project
git pull --ff-only origin main
source scripts/cluster_env.sh
df -h .
sbatch --time=2-00:00:00 scripts/submit_critics.sh --full
```

This requests one A5000 for up to 48 hours. Queue time is separate. The previous
1,536-answer run took 43 minutes; 80,000 answers is about 52 times that volume,
before accounting for longer fitting or differences in answer length. Plan for
roughly two days, not another 43-minute run. The time request is a limit, not a
completion guarantee. If Slurm rejects that limit, use the permitted limit and
resume afterward.

Full consecutive-state caches can use roughly 75 GB at 300 tokens per answer,
and around 500 GB if every answer reaches the 2,048-token cap, plus other files.
Check available storage before starting. No dependency changes are required.

Replace `JOB_ID` with the number returned by `sbatch`:

```bash
squeue -j JOB_ID
cat outputs/critics-JOB_ID/status.json
tail -n 30 outputs/critics-JOB_ID/generate.log
tail -n 30 outputs/critics-JOB_ID/extract.log
tail -n 30 outputs/critics-JOB_ID/fit.log
```

Files appear as their phases begin. When finished:

```bash
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed
cat outputs/critics-JOB_ID/report.txt
```

For a failed phase, read its log and `slurm-critics-JOB_ID.out`. If a job times out,
wait until it has stopped, then resume into its original directory:

```bash
sbatch --time=2-00:00:00 scripts/submit_critics.sh --full \
  --out outputs/critics-OLD_JOB_ID
```

Raw answers are now saved in `pending/` before scoring. A scoring failure leaves
them available for diagnosis and reuse; a successful scored batch replaces them
in `trajectories/`. Completed question shards and selected heads are reused; an interrupted head's
tuning restarts. Resuming requires identical code, model, verifier, runtime and
settings. Do not point this new protocol at the old fixed-prefix experiment or
run two jobs against the same output directory. Keep code unchanged during a run.

The report will tell us whether small nonlinear heads improve on a constant,
whether direct linear regression is competitive, and whether fitting hit its
budget. Review reward examples in `review.txt` before drawing conclusions.

## Answer failures and resuming outputs/critics-12654337

The original job saved 408 batches before answer extraction raised `SympifyError`.
Resume job 12658994 later failed on a per-answer symbolic verification timeout.
The critic experiment now excludes unverifiable answers rather than assigning
them incorrect labels. Only `correct` and `incorrect` statuses with nonempty
generated tokens enter feature extraction and fitting. Unparseable expressions,
timeouts, overlong inputs, parser/verification errors and other per-answer grader
exceptions are saved with a null score and skipped. Genuine verified wrong answers
still receive 0. Historical failed-status rows may retain their old numeric score
on disk, but their status excludes them from fitting as well.

A whole-batch verifier timeout triggers individual retries; a response whose
worker still times out is excluded. Invalid references, missing dependencies,
memory/I/O failures and crashed workers remain fatal. GPU failures and invalid
model/configuration state also remain fatal; ignoring these cannot produce a
valid dataset. A split with no usable answers cannot be fitted.

This mode is specific to the frozen critic experiment. PPO/GRPO continue to use
the numeric reward hook; they are not passed null rewards.

Reports include retained/excluded counts, exclusion rates by split, actual
retained question counts and verifier status counts. `excluded-responses.json`
contains excluded answers; `verifier-timeouts.json` lists timeout cases separately.
During generation, statuses and details are already saved in `trajectories/`.
Metrics are conditional on the retained, verifiable answers. Exclusion can select
an easier subset, so inspect coverage before making accuracy claims. The
three-second per-answer limits have not been increased.

After publishing and pulling this fix, migrate the stopped run on the login node:

```bash
python3 scripts/migrate_critic_parser.py outputs/critics-12654337
sbatch --time=2-00:00:00 scripts/submit_critics.sh --full \
  --out outputs/critics-12654337
```

The migration checks known prior hashes for affected code, unchanged remaining tracked files and questions, and saved
batch structure. It refuses unrelated changes. It backs up the prior manifest
as `manifest.before-exclusion-fix.json` and writes `exclusion-fix-migration.json`,
retaining earlier fix history. It prints the current preserved count. It applies
only to stopped generation-only runs with no feature/fitting caches yet.
Existing answers and scores are unchanged. Normal resume checks still apply.
The new Slurm job writes its log under its new ID; experiment outputs stay in
`outputs/critics-12654337`. Keep code unchanged while the resumed job runs.
