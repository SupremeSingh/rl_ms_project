# Nonlinear value-head benchmark

This is supervised prediction of eventual correctness from saved Numina prefixes,
not an RL run. It compares three models on exactly the same data:

| Model | Updated parameters |
|---|---|
| Existing SLICE + Ridge | Fresh Ridge weights, epsilon 0.01 |
| Frozen SLICE + two-layer MLP | MLP only |
| Joint SLICE + two-layer MLP | Compressor projection/bias and MLP |

Qwen is never loaded: its selected-layer RMS-normalized vectors are already cached.
Input centering and scaling remain fixed at their saved offline values. There is
no output LayerNorm in any arm. The original sparse-768 filter is copied; source
checkpoints, old results and the online PPO pipeline are never changed.

The neural path is:

```text
Cached five-block vectors (7,680), already RMS-normalized per block
    → saved training-only centering/scaling (fixed)
    → SLICE: row-normalized learned projection + bias → ReLU (768)
    → Linear(768, 256) → ReLU → Linear(256, 1) → sigmoid
    → probability of eventual correctness
```

In the joint arm, gradients flow through the MLP and compressor, including the
projection-row normalization. Only the old temporary SLICE training readout is
unused/frozen. ReLU is differentiable almost everywhere. The frozen arm uses the
same initial compressor and MLP weights and minibatch order as its paired joint
arm for each seed. Three MLP initialization/shuffle seeds (42, 43, 44) share one
previously selected compressor; this is not three independent representation fits.

Both neural arms minimize **mean squared prediction error + 0.01 × mean SLICE
activation**. In the frozen arm the activation penalty is constant and has no
effect on gradients; in the joint arm it retains the existing sparsity preference.
Thus the joint comparison includes that regularization choice. It is not BCE
training, and no claim of globally optimal nonlinear fitting is made.

Defaults: AdamW, head learning rate 1e-3, compressor rate 1e-4, weight decay 0.001,
gradient norm clipping at 1, batch size 2,048, at most 30 epochs, and early stopping
after five epochs without a new best validation Brier. The lowest validation-Brier
checkpoint is restored. Test scores are computed only after all six neural fits
are complete. No seed or arm is selected using test results.

Ridge fits the same sampled prefixes directly to their final rewards, using its
existing unbounded readout. MLP outputs are sigmoid-bounded; raw and clipped Ridge
Brier are both reported so that output clipping is not mistaken for an MLP gain.
This is a benchmark of complete predictors, not a pure test of head depth alone.

Outputs include validation/test Brier and threshold-0.5 accuracy, per-seed
results, mean and SD of neural test Brier, retained class frequencies, paired
seed/question bootstrap contrasts, learning histories, best model checkpoints,
and per-prefix predictions/labels. Intervals are exploratory and unadjusted.
Verifier exclusions and the original inspected question splits are reused. The
compressor was previously validation-selected, so this is not a new benchmark.
Runtime records are elapsed times for each neural fit, including validation and
checkpointing; they do not include historical offline feature generation.

## Duke command

After pushing these files from your computer:

```bash
cd /usr/xtmp/ms785/rl_ms_project
git pull --ff-only origin v2
source scripts/cluster_env.sh
mkdir -p logs

sbatch tools/diagnostics/submit_slice_mlp.sh \
  outputs/jlens-large-numina-v2 \
  outputs/numina-no-folds-v1 \
  outputs/slice-mlp-v1
```

The historical source-directory name is only a path: no J-lens is used. The job
requests **one A5000**, eight CPU cores and 64 GB RAM. Twelve hours is a scheduler
ceiling, not a measured ETA. The six neural fits run sequentially within one job.
No generation, encoder extraction, package reinstall or PPO calibration is needed.

```bash
tail -n 30 logs/slice-mlp-JOB_ID.out
cat outputs/slice-mlp-v1/status.json
cat outputs/slice-mlp-v1/report.txt
```

The script verifies source/model identities and cached-file hashes before fitting
and records its protocol. If interrupted, the same command reuses completed
validation fits under the identical code/config/runtime; an incomplete neural fit
restarts from its original initialization. It does not resume an optimizer midway
through an epoch. Keep code and inputs unchanged when retrying. A changed protocol
requires a new output directory.

Best checkpoints contain the entire feature-normalization/compressor/MLP state.
They are research checkpoints loaded with the `ValueMLP` class in the benchmark,
not published `Critic.load()` artifacts and not PPO filter exports. To later use
an improved joint compressor with Ridge/LSTD, first freeze it and rebuild all
matrix statistics in its new feature coordinates. Do not reuse old `A,b`.
