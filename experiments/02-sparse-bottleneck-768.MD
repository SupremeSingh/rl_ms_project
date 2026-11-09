# Does a learned 768-feature bottleneck preserve useful correctness signals?

**Yes: sparse 768 + Ridge reached Brier 0.18505 and 72.66% offline prediction accuracy, versus 0.20216 Brier for matched-width PCA. Plain and sparse compressors were close; sparsity did not establish a hardware speedup.**

| Representation + Ridge | Accuracy | Brier |
|---|---|---|
| Plain bottleneck | 72.63% | 0.18529 |
| Sparse bottleneck | 72.66% | 0.18505 |
| Full normalized multi-layer | 72.01% | 0.18917 |
| Two-thirds layer | 70.27% | 0.19702 |
| Final layer | 66.89% | 0.21306 |

Three compressor seeds (42–44); frozen Qwen features; BCE-supervised compressors; all-token Ridge and LSTD(.99). Sparse training adds an activation penalty to a learned row-normalized linear projection followed by ReLU. This is inspired by sparsification, not a replication of the full rate-reduction/CRATE architecture.

The 768-feature study is separate from the other bottleneck width: small differences between study means are not proof of a width advantage. The 768-wide choice was a compact practical baseline. Kernels and matrix accumulation remain dense.

## Evidence and scope

[Full report extract](data/sparse768-report.txt). Cluster: `outputs/numina-sparse-768-v1/report.txt`. Offline exploratory prediction on inspected Numina questions; no PPO or total-cost claim.
