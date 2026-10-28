# One shared cumulative critic (v2)

All questions use one persistent A, b and w. There is no question splitting,
held-out fold or opposite-head prediction. The conceptual assignment is always
`f = 1`; Python indexes the single system as 0. The algorithm is:

```text
Initialize A = 0, b = 0; freeze encoder, bottleneck and normalization.
For each PPO update:
    Generate and score 64 fresh answers.
    Add every answer's Ridge/LSTD contributions to the same A and b.
    Solve (A + 0.01 I) w = b once.
    Predict every current prefix with value(x) = dot(w, x).
    Compute GAE; update the actor with PPO on the fresh batch.
    Retain A and b for the next update.
```

This is fit-then-predict on the current batch, exactly as requested. Gamma is 1,
LSTD lambda 0.9, GAE lambda 0.95; traces reset per answer; terminal successor
features are zero, including the intercept. No buffer, normalization by sample
count, encoder/compressor refresh or delayed critic fit. Diagnostics from this
head are in-sample, not held-out value estimates. Historical-policy statistics
remain uncorrected.

## Same original comparison, one head in each linear arm

The six methods are unchanged: sparse-768 Ridge PPO, sparse-768 LSTD PPO,
two-thirds Ridge PPO, two-thirds LSTD PPO, conventional PPO and GRPO. Every
Ridge/LSTD arm explicitly uses `critic_folds=1`; no two-fold control runs are
submitted. Conventional PPO and GRPO retain their original implementations.

Keep the original five actor seeds, 120 updates x 64 answers, 2,000 test questions,
frozen data/filter, PPO profile and eight primary method comparisons. All 30
actors start from the base model, with empty critic statistics. Comparison to
previous runs is historical, not a contemporaneous fold ablation. Old results
and directories are preserved. The library retains two-fold compatibility for
reproduction of those old studies; the new protocol rejects any linear arm
configured to use two folds.

## Commands after pushing v2

From the cluster checkout, with no jobs using it while switching/installing:

```bash
git switch v2
git pull --ff-only origin v2
source scripts/cluster_env.sh
srun -p compsci --cpus-per-task=8 --mem=128G --time=00:45:00 --pty bash -i
```

Inside the allocation:

```bash
source scripts/cluster_env.sh
bash scripts/container_exec.sh bash scripts/install_critic_package.sh
bash scripts/container_exec.sh "$PWD/.venv/bin/python" \
  scripts/check_slice_release.py --out outputs/slice-no-fold-release-v1
cat outputs/slice-no-fold-release-v1/verification.json

bash scripts/container_exec.sh "$PWD/.venv/bin/python" -m math_rl \
  --stage prepare-no-fold-study \
  --source outputs/numina-prod-replay-v1 \
  --out outputs/numina-no-folds-v1
exit
```

Require passed=true and package version 0.3.0. Preparation copies verified
inputs, never trained actors or old matrix statistics. It records the new code
and installed runtime; do not edit inputs afterward. No compressor retraining or
new PPO calibration is performed.

From the login node:

```bash
source scripts/cluster_env.sh
bash scripts/submit_numina_online.sh \
  --out outputs/numina-no-folds-v1 \
  --full --smoke-first --concurrency 15
cat outputs/numina-no-folds-v1/jobs.tsv
squeue -u "$USER"
```

The six-arm GPU smoke must pass before the 30-run array starts. It checks one
system per linear arm, retained counts, saved-head solves and conventional PPO
health. Fifteen concurrent runs require up to 30 GPUs. Prior runtimes suggest
roughly 16-24 hours of execution including smoke, but queue delays and hardware
availability can extend that; there is no guaranteed deadline.

```bash
cat outputs/numina-no-folds-v1/smoke.json
# After the report completes:
cat outputs/numina-no-folds-v1/report.txt
cat outputs/numina-no-folds-v1/summary.json
```

Use fresh output names if a directory exists. If an earlier ablation chain was
already submitted, cancel that specific chain before starting this one; do not
run the obsolete prepare-fold-study command. A submission timeout can still
create a job: inspect jobs.tsv and Slurm before retrying.
