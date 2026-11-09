# Does a learned 1536-feature bottleneck preserve useful correctness signals?

**Yes: sparse 1536 + Ridge reached Brier 0.18489 and 72.71% offline prediction accuracy, versus 0.19773 Brier for matched-width PCA. Plain and sparse compressors were close; sparsity did not establish a hardware speedup.**

| Representation + Ridge | Accuracy | Brier |
|---|---|---|
| Plain bottleneck | 72.54% | 0.18532 |
| Sparse bottleneck | 72.71% | 0.18489 |
| Full normalized multi-layer | 72.01% | 0.18917 |
| Two-thirds layer | 70.27% | 0.19702 |
| Final layer | 66.89% | 0.21306 |

Three compressor seeds (42–44); frozen Qwen features; BCE-supervised compressors; all-token Ridge and LSTD(.99). Sparse training adds an activation penalty to a learned row-normalized linear projection followed by ReLU. This is inspired by sparsification, not a replication of the full rate-reduction/CRATE architecture.

The 1536-feature study is separate from the other bottleneck width: small differences between study means are not proof of a width advantage. The 768-wide choice was a compact practical baseline. Kernels and matrix accumulation remain dense.

## Evidence and scope

[Full report extract](data/sparse1536-report.txt). Cluster: `outputs/numina-sparse-1536-v1/report.txt`. Offline exploratory prediction on inspected Numina questions; no PPO or total-cost claim.
