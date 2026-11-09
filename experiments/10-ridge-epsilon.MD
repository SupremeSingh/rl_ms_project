# Does stronger Ridge regularization materially improve the LayerNorm variant?

**Sweeping epsilon from .1 to 100 greatly improved SLICE + LayerNorm conditioning, but its displayed Brier stayed 0.18302 and accuracy stayed 73.2%. This was a conditioning improvement, not evidence of a meaningful prediction or PPO improvement.**

| epsilon | LayerNorm Brier | LayerNorm condition | Baseline SLICE Brier |
|---|---|---|---|
| 0.01 | 0.18302 | 5.23e9 | 0.18346 |
| 0.1 | 0.18302 | 5.23e8 | 0.18346 |
| 1 | 0.18302 | 5.23e7 | 0.18346 |
| 10 | 0.18302 | 5.23e6 | 0.18346 |
| 100 | 0.18302 | 5.23e5 | 0.18344 |

Same offline sampled-prefix data as the normalization study; no hyperparameter selection on these test results. A declared scale-relative policy also produced approximately unchanged predictions (LayerNorm epsilon ≈63; baseline epsilon ≈57.9). Tiny unrounded differences exist even when the table rounds identically.

LayerNorm’s own epsilon was 1e−5; that is separate from Ridge’s epsilon. We use .01 as a simple default for the established cumulative recipe, not a universal optimum. Later transfer experiments selected much larger penalties on validation. Matrix sums and counts determine the effective penalty; changing data scale can change its appropriate value.

## Evidence and scope

[Full epsilon sweep](data/epsilon-report.txt). Cluster: `outputs/ridge-normalization-epsilon-v1/report.txt`. No RL or general-purpose optimal-epsilon claim.
