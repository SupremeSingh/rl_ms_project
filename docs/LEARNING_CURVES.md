# Five-method learning curves with independent actor jobs

**Question: how do actor accuracy and across-seed variation evolve with sampled
answers and measured training compute?** No learning-curve results exist yet.

## Locked design

- Five arms: SLICE/token, SLICE/chunk, conventional PPO, GRPO and RMS-normalized
  two-thirds layer + Ridge/token. Three **fresh seeds 1201–1203**: 15 actors.
- Original completion prompts, frozen base/filter and normalization, one
  cumulative head, epsilon .01, gamma 1, GAE .95; PPO retains `higher_lr` and
  GRPO its original estimator. **No fixed-action replay or benchmark runtime.**
- 120 updates × 64 fresh answers/run. Monitoring is the full existing **validation**
  split at updates **0, 10, …, 120**; no monitoring-based selection or early stopping.
- Five independent smoke jobs use seed 19, two real updates each, the smaller
  prepared smoke data, and monitoring/checkpoint saving every update.
- Main training uses **15 independent Slurm array tasks**, each with two A5000s,
  capped at **six concurrent actors (12 GPUs total)**. A fixed shuffled task map
  covers every method/seed once. Placement records retain each actor's node,
  assigned GPUs and Slurm job ID. Methods can run on different nodes, so measured
  time curves are hardware-sensitive; the separate controlled benchmark remains
  the evidence for computational overhead under matched hardware/workloads.
- Keep **every actor HF checkpoint at updates 10–120**, not only the last one.
  Step zero is the pinned base model. Only the latest distributed critic snapshot
  is retained; this is not a promise of resumable optimizer checkpoints.
- After all training completes, a separate gate verifies curve coverage and
  retained actors. All 15 final actors plus a fresh base are then evaluated
  greedily on the same 2,000-question test split: **32,000 test answers**.
  Intermediate checkpoints are not tested or selected by test outcomes.

## Time and plots

Driver hooks time monitoring and checkpoint operations separately, without
changing the advantage estimators, workers, PPO losses, actor sampling or update
budgets. Training time includes fresh generation, rewards, feature transport,
critic fitting and actor updates; measured monitoring and checkpoint intervals
are excluded from the training GPU-hour axis and retained in each run's ledger.
Initialization and outer base/final evaluations are separate. This is measured
elapsed time × two allocated GPUs, not utilization or a full allocation ledger.

Each actor writes `learning-curve.jsonl`: step, monitoring accuracy, evaluation
duration and cumulative training GPU-hours. `training-clock.json` records the
fit/evaluation/checkpoint time totals. These direct files do not rely on Ray
forwarding the final console line. The gate requires 13 points in order and
rejects missing/duplicate points. The duplicate outer final validation is kept
as an audit record but excluded from curve aggregation.

`learning-curves.png`/`.pdf` show mean ± **sample seed SD** versus updates and
training GPU-hours. For time curves, validation observations are linearly
interpolated onto a common time grid within each method's three-seed coverage;
there is no extrapolation past the earliest finishing seed. This is interpolation
between measured evaluations, not finer-grained accuracy measurements.

The final summary reports test means/seed SD and two predeclared adjusted paired
seed/question bootstrap contrasts: SLICE/chunk minus SLICE/token and minus PPO.
Three seeds limit precision. Monitoring and test data have been inspected in
prior work; this is not an independent benchmark or proof of universal stability.

## Storage and runtime

Retaining 180 HF actor snapshots can require **roughly 0.6–1.2 TB**, depending on
export precision. Preparation uses the larger actual base/export size from the
prior value-repeat checkpoint when available (otherwise assumes FP32 exports),
times 180 snapshots plus 40% headroom, and fails if free space is insufficient. Smoke
checks that HF-only exports and retention really work before the main study.
Do not delete source studies to satisfy this check automatically.

The default wall limit is **12 hours per actor**. With six available allocations,
15 actors require about three scheduling waves; prior actor runs took roughly
3–5.5 hours before the added monitoring/export overhead. Allow about a day or
longer including queues, checkpoint exports and final evaluation. This is an
estimate, not a completion guarantee; check the partition's wall limit.
No partial training is automatically resumed or outcome-based replacement made.
Each method has the same evaluation/export schedule, but HF export overhead is
still an operational cost recorded separately from the learning-efficiency axis.

## Commands

After pushing the new files to v3, and once other jobs using the checkout finish:

```bash
cd /usr/xtmp/ms785/rl_ms_project
git switch v3
git pull --rebase origin v3
source scripts/cluster_env.sh
df -h .
scontrol show partition compsci-gpu

python3 scripts/submit_learning_study.py --mode smoke \
  --source outputs/numina-no-folds-v1 \
  --out outputs/numina-learning-curves-parallel-v1 --concurrency 6
```

Inspect the submitted jobs and require a passed gate:

```bash
cat outputs/numina-learning-curves-parallel-v1-submission/jobs.tsv
squeue -u "$USER"
cat outputs/numina-learning-curves-parallel-v1/smoke.json
```

After smoke passes, review its logs before main submission. Confirm the
partition permits a 12-hour allocation per actor.

```bash
python3 scripts/submit_learning_study.py --mode main \
  --out outputs/numina-learning-curves-parallel-v1 \
  --concurrency 6 --hours-per-actor 12
```

15 independent actor jobs → training gate → final test array → report run automatically
with successful-job dependencies. Freeze code/environments throughout. The
final outputs are `report.txt`, `summary.json`, `verification.json` and
`learning-curves.png`/`.pdf` under the study directory. Existing actors and
studies are never overwritten. GPU integration must pass on the cluster;
local CPU tests do not establish successful distributed training.
