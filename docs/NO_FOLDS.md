# One-head cumulative fitting (v2)

This experiment removes question cross-fitting without changing the learned
sparse-768 representation or PPO recipe. `folds=1` means one shared matrix
system, not one train/test split. Each batch is added before solving and predicting
that batch. No statistics are thrown away, normalized or reset. The two-fold
control retains the previous opposite-question prediction rule.

## Matched comparison

| Arm | Head fitting | Value prediction |
|---|---|---|
| `ridge-fold1` | All historical/current answers; one Ridge system | Same shared head |
| `ridge-fold2` | Permanent question groups; two Ridge systems | Opposite fold |
| `lstd-fold1` | All historical/current answers; one LSTD system | Same shared head |
| `lstd-fold2` | Permanent question groups; two LSTD systems | Opposite fold |

Five actor seeds (301-305), 120 updates x 64 answers, final greedy evaluation on
the same 2,000 inspected questions. Two predeclared primary contrasts: no-fold
minus two-fold accuracy for Ridge and for LSTD, with paired seed/question
bootstrap intervals and Bonferroni adjustment over those two comparisons.
This is an ablation on reused questions, not a fresh benchmark.

All four arms use frozen base encoder/compressor, FP32 feature capture, gamma 1,
LSTD lambda 0.9, GAE lambda 0.95, raw-sum epsilon 0.01 (including intercept), no
buffer, fresh actor rollouts and zero initial statistics. The only arm setting
changed within each Ridge/LSTD pair is `critic_folds`. Effective regularization
per fitted sample consequently differs with the per-head sample count; epsilon
is deliberately not retuned. No-fold diagnostics are explicitly in-sample and
must not be interpreted as held-out value quality. Neither mode corrects
historical-policy mismatch or guarantees stable actor training.

## Run on Duke

Push the finished `v2` changes from your Mac first. Run from the existing cluster
checkout, with no jobs using that checkout while switching/installing:

```bash
git switch v2
git pull --ff-only origin v2
source scripts/cluster_env.sh
srun -p compsci --cpus-per-task=8 --mem=128G --time=00:45:00 --pty bash -i
```

Inside the CPU allocation, install internal package 0.3.0, verify the wheel and
capture unchanged data/filter/model inputs from the completed production replay:

```bash
source scripts/cluster_env.sh
bash scripts/container_exec.sh bash scripts/install_critic_package.sh
bash scripts/container_exec.sh "$PWD/.venv/bin/python" \
  scripts/check_slice_release.py --out outputs/slice-folds-release-v1
cat outputs/slice-folds-release-v1/verification.json

bash scripts/container_exec.sh "$PWD/.venv/bin/python" -m math_rl \
  --stage prepare-fold-study \
  --source outputs/numina-prod-replay-v1 \
  --out outputs/numina-folds-v1
exit
```

Require `passed: true` and package version 0.3.0 before proceeding. Use new output
names if these directories exist. Preparation copies no actors or value heads;
it locks the new code/runtime provenance, original data and compressor hashes.
There is no offline refitting, conventional PPO arm or additional PPO calibration.

From the login node, submit the four-arm smoke, then the gated 20-run array and
CPU report:

```bash
source scripts/cluster_env.sh
bash scripts/submit_numina_online.sh \
  --out outputs/numina-folds-v1 \
  --full --smoke-first --concurrency 10
cat outputs/numina-folds-v1/jobs.tsv
squeue -u "$USER"
```

At most 20 GPUs run concurrently (two per training job). Based on the prior
5-8-hour runs, allow roughly 14-22 hours of elapsed execution including the
sequential smoke, subject to available hardware and queue delays. This is not a
guaranteed completion deadline. Submission errors stop the chain; a scheduler
timeout can still create a job, so inspect accounting before retrying.

```bash
cat outputs/numina-folds-v1/smoke.json
# After the report job finishes:
cat outputs/numina-folds-v1/report-0-status.json
cat outputs/numina-folds-v1/report.txt
cat outputs/numina-folds-v1/summary.json
```

Smoke requires two actual PPO updates for every arm and audits the number of
matrix systems, retained answer counts, head routing, finite weights and agreement
between the final saved head and its accumulated system. Full reporting repeats
the checkpoint checks over all 120 updates and preserves generated test answers
in `audit.jsonl`. CPU tests establish the equations and orchestration; real-model
GPU integration still needs this new smoke. Historical two-fold results are not
relabelled as no-fold results.
