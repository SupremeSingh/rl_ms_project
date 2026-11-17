# Encoder comparison paused: 17 November 2026

Decision: stop further encoder-study compute for now and focus on other comparisons.
No PPO has run in this study. No automatic recovery jobs should be submitted.

The experiment README and pause rationale are in
[Experiment 23](../experiments/23-encoder-critic-paused.MD). The cluster queue was
confirmed empty after cancellation. The validation report was saved in
`outputs/encoder-paused-validation-v1/`.

All 104,000 actor answers, three encoder adaptations, and 416,000 feature files
completed. SLICE C/D fitting exceeded the 64 GB host RAM allocation. Direct A/B
fits succeeded. Available validation raw Brier:

| Readout | Causal | Adapted-42 | Adapted-43 |
|---|---:|---:|---:|
| A, final mean | 0.1526351109 | 0.1721746055 | 0.1707720234 |
| B, two-thirds mean | 0.1487123081 | 0.1659588358 | 0.1649739875 |

These are preliminary validation results, not test or MC-reference results. The
encoder loses both readouts in both completed adaptation seeds. Adapted-44 has no
completed fits in the pause report. C/D have no completed comparison yet.

## Stop on the cluster

Cancel the remaining fitting and downstream jobs; do not delete any study files:

```bash
scancel 13771038 13771039 13771040
squeue -u "$USER" -o '%.22i %.30j %.12T %.30R'
```

If an interim analysis was submitted separately, its job IDs are different. Inspect
its `fit-*.log` / submission output and cancel those specific job IDs as well.
Cancellation stops this experiment's remaining work; it is not Slurm suspension.
Original successful artifacts and partial failed directories remain intact.

## Save the available results (no fitting)

The new scripts must first be copied to the cluster. From the repository root:

```bash
bash scripts/container_exec.sh "$PWD/.venv-encoder/bin/python" -u \
  tools/encoder/recover_compare.py --stage validation \
  --study outputs/encoder-vine-unpadded-numina-v1 \
  --out outputs/encoder-paused-validation-v1
```

This reads only completed A/B checkpoint files, verifies their seals, and writes
`report.txt` and `summary.json` in a fresh separate directory. It reads no test
features and generates no new answers. Use a fresh output name if rerunning.

## Recovery implementation (available for later, not part of the pause)

`recover_compare.py` uses disk-backed sampled training matrices, avoiding the
retained list plus concatenated tensor. Normalization computes the same global
centered scalar RMS in bounded blocks with FP64 accumulation. Roundoff can differ
from the original reduction. The architecture, data, sample seeds, sparsity loss,
optimizer, epoch budget, Ridge fit and validation checkpoint selection are retained.
At most about 20 GiB scratch per C/D task is needed for the sampled matrix; the
runner prefers `SLURM_TMPDIR` when available. Keep two-task concurrency initially.

Only new scripts are introduced: the original runtime files and sealed manifest
are not edited. Recovery records its script hash and the original runtime identity.
It validates the original study at execution. If original runtime code has changed
since extraction, restore the pinned code in a separate checkout before recovery;
do not edit the manifest's runtime hashes.

After all original/interim jobs have stopped, prepare a new output:

```bash
python3 tools/encoder/recover_compare.py --stage prepare \
  --study outputs/encoder-vine-unpadded-numina-v1 \
  --out outputs/encoder-vine-unpadded-numina-recovery-v1
python3 tools/encoder/submit_recovery.py \
  --out outputs/encoder-vine-unpadded-numina-recovery-v1
```

Preparation verifies all extraction seals and completed fits, links inputs and
successful fits, and lists missing tasks. Failed source directories are neither
deleted nor reused. The submit command above only prints a plan. Nothing runs.

Only when this research direction is explicitly resumed:

```bash
python3 tools/encoder/submit_recovery.py \
  --out outputs/encoder-vine-unpadded-numina-recovery-v1 --submit
```

It submits missing fits, missing MC shards, then evaluation with fresh dependencies.
The submission ledger prevents duplicate submissions. A partial recovery fit is
never overwritten or silently retried. Original partial MC shards are recomputed
in the fresh output; sealed completed MC aggregates are reused. Keep original files
available: recovery uses symlinks, not archival copies. This is a recovery of an
inspected experiment, not a new confirmatory test set.
