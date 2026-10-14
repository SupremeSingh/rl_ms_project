# Compressing complete multi-layer J-lens readouts

This follow-up reuses a completed large J-lens study. It tests whether full J-lens
readouts preserve value-prediction information better than matched hidden-vector
controls at a smaller dimension. It does not generate answers, run an encoder,
change the source study, or perform PPO updates.

## Comparison

Fit centered PCA separately on each representation's **all-token training
covariance**, using its existing fixed isotropic normalization:

- Complete J-lens concatenation.
- Raw hidden concatenation at the same five layers.
- RMS-normalized hidden concatenation at those layers.

Retain 768, 1,536 and 3,072 components. No whitening or additional rescaling is
applied. These are dense projections of the full features, not top-word sparse
reconstructions. The uncompressed versions and the original two-thirds/final
single-layer critics remain controls.

Fit Ridge and LSTD(0.99), preserving gamma=1, raw sums and the source epsilon
(0.01 in the completed study, including the intercept). PCA uses no reward labels.
Its state weighting matches all-token linear fitting, so longer answers contribute
more states. This differs from fitting PCA on the sampled-prefix neural dataset.

For each arm and method, validation Brier chooses among the three dimensions and
the uncompressed candidate. All selections finish before test loading. Report all
predefined candidates, not just the best-looking test score. Test questions were
already inspected in the previous study: comparisons and unadjusted paired
question intervals are exploratory. Existing conditional verifier labels remain;
this job does not audit or repair them.

## Why no encoder pass is needed

Let the source design be `phi=[normalized_features, 1]`. A fixed affine PCA map
has a matrix T such that `compressed_phi = phi @ T`. Then:

```
A_compressed = T.T @ A @ T
b_compressed = T.T @ b
G_compressed = T.T @ G @ T
c_compressed = T.T @ c
```

Eligibility traces commute with this fixed transformation. Terminal zero remains
zero. Apply epsilon AFTER projection, rather than projecting an already
regularized system. PCA covariance itself comes from the training Gram matrix's
feature block and intercept column, so the original token features need not be
re-extracted. Tests compare this shortcut with explicit trajectory reaccumulation.

The small recentering is learned from the all-token training distribution. The
source normalization was fitted on training sampled prefixes; neither uses
validation/test data. The PCA transform and source preprocessing are saved and
must stay fixed for any future cumulative LSTD use.

## Run

From the repository root on the cluster:

```bash
source scripts/cluster_env.sh
sbatch scripts/submit_jlens_compression.sh outputs/jlens-large-numina-v2
```

The job requests 8 CPU cores, 96 GB RAM and a 12-hour limit; no GPU. The time limit
is a ceiling, not a runtime prediction. It performs dense PCA decompositions and
matrix projections but avoids the expensive generation and extraction stages.

Outputs go to `outputs/jlens-compression-JOBID/`:

- `report.txt`: all candidate scores, validation selections and paired differences.
- `summary.json`: metrics, retained variance and comparisons.
- `selection.json`: validation-selected dimensions (never selected on test).
- `protocol.json`: source/code hashes and fixed settings.
- Per-arm `.pt` files: PCA components, normalization, fitted weights, numerical
  checks, validation metrics and fitting wall time.
- `test-predictions.npz`: predictions and matching question IDs/labels.
- `status.json`: running, complete or the failure message.

To resume, use the same output path after the first job has stopped:

```bash
sbatch scripts/submit_jlens_compression.sh outputs/jlens-large-numina-v2 \
  --out outputs/jlens-compression-JOBID
```

Completed arm checkpoints are hash checked and reused. Changing the protocol,
source hashes or tracked implementation requires a new output directory.

## Interpreting cost

Relative to 7,680 features, a 1,536-dimensional system uses approximately 4% of
the matrix storage and 0.8% of conventional dense-solve work; 3,072 uses 16% and
6.4%. These are scaling estimates, not measured full-training savings.

Online extraction would still need the five selected layers and lens operations.
This offline evaluation folds the projection into weights to avoid large temporary
feature caches; its prediction runtime is not a benchmark of a deployed compressed
critic. Measure encoder, projection, statistic construction and solve costs before
claiming a cheaper PPO system. High retained variance alone does not establish
preserved value quality: use held-out Brier, accuracy and calibration.
