# Does adding LayerNorm after SLICE improve offline Ridge predictions?

**It improved raw Brier slightly, from 0.18346 to 0.18302, with both accuracies rounding to 73.2%. The regularized condition number rose from 3.04×10⁴ to 5.23×10⁹ at epsilon .01; no PPO benefit was tested.**

| Ridge representation, epsilon .01 | Raw Brier | Accuracy | Condition number |
|---|---|---|---|
| RMS two-thirds | 0.18957 | 71.8% | 5.36e5 |
| SLICE baseline | 0.18346 | 73.2% | 3.04e4 |
| SLICE + output LayerNorm | 0.18302 | 73.2% | 5.23e9 |

Same saved sampled prefixes and labels; 2,000 inspected test questions; fixed previously validation-selected compressor. No encoder or compressor retraining, calibration or RL. The question-mean raw-Brier difference for LayerNorm minus baseline was −0.0004136, exploratory interval [−0.0007227, −0.0001127]. This differs from the prefix-weighted Brier values in the table.

Current SLICE already has per-block RMS normalization and training-only centering/scalar scaling before its projection. Output LayerNorm is an additional transform. Centering output coordinates constrains their sum; with an intercept, that creates a redundant feature direction, so worse conditioning can coexist with similar predictions. We retained the simpler no-output-LayerNorm baseline. This study compared RMS normalization of the raw layer, not a separate raw-layer LayerNorm arm, and did not test LayerNorm with LSTD.

## Evidence and scope

Cluster: `outputs/ridge-normalization-v1/report.txt`. The same baseline rows and original comparisons are preserved in the [epsilon follow-up extract](data/epsilon-report.txt). Offline reused data; no fresh-benchmark or numerical-failure claim.
