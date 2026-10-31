# Offline normalization comparison

This is correctness prediction with Ridge, not PPO. Reuse the saved Numina
sampled-prefix arrays and the exact frozen sparse-768 filter from an online-study
manifest. No new answers, encoder inference, compressor fitting, lens computation
or actor updates are performed. Existing online features and artifacts are unchanged.

## Three representations

1. **Normalized two-thirds:** divide each 1536-coordinate raw block vector by
   `sqrt(mean(h**2) + model_rms_epsilon)`. No learned gain or dataset centering.
2. **SLICE + LayerNorm:** unchanged frozen SLICE output, followed by per-example
   LayerNorm with epsilon `1e-5`, no learned affine parameters. Applied after ReLU,
   before appending the intercept. This changes sparsity and removes the vector's
   common offset; it is not whitening and need not improve prediction.
3. **SLICE baseline:** unchanged frozen SLICE features, without output normalization.

The same cached questions, answers, prefix lengths and correctness labels are used
in all arms. The cache already contains per-block RMS-normalized concatenation for
SLICE; that normalization is not applied twice. The filter's existing training-only
centering/scaling remains unchanged. Two-thirds features are taken from the raw cache.
All heads are newly fitted on **sampled prefixes**, not the historical all-token
statistics. The baseline reproduces the representation, not historical accuracy.

## Predeclared regularization

Every arm fits both policies, with an intercept appended and penalized:

- Primary: `epsilon = max(0.01, 0.0001 * trace(A_train) / design_width)`.
- Legacy control: fixed `epsilon = 0.01`.

The relative policy is a priori diagonal loading, based only on the training Gram
matrix. It scales with feature magnitude and training size; it does not guarantee
a target condition number or an optimal predictor. Epsilon values, condition
numbers, residuals and weight norms are reported. No policy or representation is
selected on test or validation scores. A future sweep would require a separate
validation selection protocol.

Raw Brier and threshold-0.5 prediction accuracy are reported for validation and
test. Clipped Brier, equal-question raw Brier, saved predictions and paired
question bootstrap intervals (2,000 draws, unadjusted) are also included. No sigmoid
is applied to raw Ridge values. Existing splits and compressor selection have
already been inspected: this is an exploratory ablation, not a fresh benchmark.
Previously retained verifier labels are reused without claiming a new reward audit.

## Run on Duke

After pushing the new files from your computer:

```bash
cd /usr/xtmp/ms785/rl_ms_project
git pull --ff-only origin v2
source scripts/cluster_env.sh
mkdir -p logs
sbatch tools/diagnostics/submit_ridge_normalization.sh \
  outputs/jlens-large-numina-v2 \
  outputs/numina-no-folds-v1 \
  outputs/ridge-normalization-v1
```

The historical source-directory name is only a path to saved **hidden-state**
arrays. Neither the tool nor the SLICE package uses J-lens for this experiment.
The job requests eight CPU cores and no GPU. Two hours is a scheduler limit, not
an ETA. Runtime depends primarily on reading/hashing the full feature cache and
matrix products. Nothing needs reinstalling if the current package already works.

```bash
cat outputs/ridge-normalization-v1/status.json
cat outputs/ridge-normalization-v1/report.txt
```

Model identity, layer ordering, cache checksums and row-level pairing are checked
before fitting. Source files are never modified. The tool refuses an existing
output directory; after a failed attempt use a new output path. These diagnostics
live outside the active training provenance glob and do not change RL behavior.
For future online normalization experiments, retrain heads from empty matrices;
do not change transforms during an accumulated-statistics run.
