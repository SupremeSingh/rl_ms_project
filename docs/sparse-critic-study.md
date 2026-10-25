# Learned sparse bottleneck for a fixed linear critic

**Active pipeline:** frozen hidden states → sparse/plain bottleneck → Ridge/LSTD.
Offline workers use `.venv-critic` (`scripts/setup_critic.sh`); PPO uses `.venv`.
The saved source directory `outputs/jlens-large-numina-v2` has a historical name:
we reuse its hidden-state arrays and trajectories, not its lens readouts. Keep
existing artifact paths unchanged for checksums and reproducibility.


This is a task-trained sparsity ablation inspired by the geometric RL paper. It
is **not** a replication of CRATE, a rate-reduction objective, an orthogonal
projection, or a guarantee of discovering a tactic manifold.

## Matched arms

Both learned arms take the saved RMS-normalized five-layer concatenation (7,680
coordinates), use the same training-only isotropic input normalization, and produce
768 ReLU features by default (`--width 1536` matches one hidden layer;
`--width 3072` matches two). Use a separate output directory for each width. They share architecture, seeds 42/43/44, learning rates
0.0003/0.001, AdamW weight decay 0.001, batch size 2,048 and 30 training epochs.

- **Plain:** binary cross-entropy on eventual answer correctness.
- **Sparse:** the same loss plus `0.01 * mean(nonnegative_features)`.

Projection rows have unit norm during the forward pass in both arms. This prevents
trivially shrinking projection weights to reduce the activation penalty; it does
not impose orthogonality. Readout weight decay also discourages compensating with
arbitrarily large output weights. Report activation zero fraction and dead units;
sparsity alone is not evidence of useful compression.

The temporary sigmoid head trains the representation. It is discarded for the
final comparison: freeze the bottleneck, then independently fit linear Ridge and
LSTD(0.99). Gamma=1, terminal-only reward, traces reset per answer, raw-sum matrices
and epsilon=0.01 including the intercept remain unchanged. No PPO update occurs.

## Stage 1: inexpensive validation screen

Use existing sampled-prefix training and validation caches, never test data.
For each trial, select the epoch by sigmoid-head validation Brier, then fit Ridge
on its frozen training-prefix features. Select the learning rate by validation
Ridge Brier independently for each arm/seed. All trials receive the same epoch
budget. Train and validation questions remain separate.

Compare the mean validation Ridge Brier of each learned arm to a newly fitted
**sampled-prefix** two-thirds-layer Ridge baseline. The full stage is recommended
only if at least one arm matches or beats that baseline. This heuristic screen is
not an inferential test and can reject a representation that would help LSTD.

Run from the repository root:

```bash
source scripts/cluster_env.sh
bash scripts/submit_sparse_critics.sh outputs/jlens-large-numina-v2 \
  --out outputs/numina-sparse-768-v1
```

Then inspect:

```bash
cat outputs/numina-sparse-768-v1/screen-report.txt
cat outputs/numina-sparse-768-v1/screen.json
```

The screen requests one GPU and a 24-hour ceiling. It generates no answers and
runs no language-model inference. The ceiling is not a runtime estimate.

## Stage 2: matched all-token Ridge/LSTD comparison

If `recommended_full` is true:

```bash
bash scripts/submit_sparse_critics.sh outputs/jlens-large-numina-v2 \
  --out outputs/numina-sparse-768-v1 --full
```

This submits four GPU statistic shards followed by a CPU report job. Each shard
makes one teacher-forced pass over its existing training answers and accumulates
all six selected frozen bottlenecks together. **Nonlinear ReLU features require
new statistics**; the PCA matrix-projection shortcut cannot compute them. Source
trajectory hashes and final answer/token counts must match the original study.

The final report includes:

- Plain and sparse bottlenecks, individual seeds and mean metrics.
- Full normalized multi-layer, two-thirds-layer and final-layer controls.
- PCA of the normalized multi-layer features at the same selected width.
- Ridge and LSTD(0.99) for every representation; matched all-token weighting.
- Brier, accuracy, calibration, raw out-of-range values and paired question
  intervals (in `summary.json`). All arms use identical retained test prefixes.

```bash
cat outputs/numina-sparse-768-v1/report-0-status.json
cat outputs/numina-sparse-768-v1/report.txt
```

Check `jobs.tsv` and `logs/` for scheduling/errors. Statistics checkpoint every
32 questions. Repeating `--full` after prior jobs stop resumes completed shards
and saved partial sums. Repeating the screen reuses completed trial checkpoints;
an interrupted trial starts again. Never submit concurrent writers to one output.

## Interpretation and limits

This is an exploratory follow-up on inspected Numina questions and inherited
verifier labels. The learned representation uses outcome supervision; PCA does
not. The plain bottleneck is essential to isolate the contribution of sparsity.
Question bootstrap intervals average over head seeds; they do not represent
independent datasets or prove superiority across models/tasks.

Training times are saved per trial and stage, plus solver residual checks. These
are offline costs, not a measured PPO speedup. The encoder pass, bottleneck
inference, feature transport and statistic construction all cost compute. Sparse
activations do not automatically produce sparse LSTD matrices or faster kernels.

Freeze both the encoder and bottleneck before any future matrix accumulation.
Changing the bottleneck requires rebuilding/resetting statistics. An online use
also needs to respect question cross-fitting when training a supervised feature
map; freezing a map trained on a question's rewards does not erase that leakage.

## Matched one-layer width: 1,536

Run a separate screen with the same data, seeds and fitting budget:

```bash
bash scripts/submit_sparse_critics.sh outputs/jlens-large-numina-v2 \
  --out outputs/numina-sparse-1536-v1 --width 1536
cat outputs/numina-sparse-1536-v1/screen-report.txt
```

When that screen recommends the all-token comparison:

```bash
bash scripts/submit_sparse_critics.sh outputs/jlens-large-numina-v2 \
  --out outputs/numina-sparse-1536-v1 --full
```

Later stages automatically recover the width from the saved protocol. An explicit
conflicting width is rejected. PCA uses the same width; the full and single-layer
controls stay unchanged. A 1,536-dimensional bottleneck gives a 1,537-square matrix
including the intercept, matching a single-layer critic's matrix dimension.
It still incurs the cost of extracting five layers and applying the bottleneck.
Doubling width approximately quadruples dense matrix storage and can increase
dense solve work eightfold. Select width on validation, not the final test scores.
The launcher also supports 3,072 dimensions, but start with 1,536 for the direct
single-layer size comparison.
