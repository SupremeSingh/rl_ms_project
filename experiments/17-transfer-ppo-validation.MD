# Did validation tuning favor stronger online Ridge regularization?

**In a one-seed, 20-update pilot, validation selected epsilon 10,000 for transferred SLICE and 1,000 for the RMS layer. The SLICE choice was at the grid boundary, so this was limited recipe selection rather than an identified optimum.**

| Candidate epsilon | SLICE validation accuracy AUC | RMS-layer validation accuracy AUC |
|---|---|---|
| .01 | .7130 | .7125 |
| 100 | .7120 | .7150 |
| 1,000 | .7125 | .7170 |
| 10,000 | .7210 | .7125 |

Eligible MATH numeric questions; actor seed 601; grid [.01, 100, 1,000, 10,000]. Validation accuracy AUC across 20 updates selected one penalty per representation before the five-seed main study. Here AUC means normalized area under the accuracy-versus-training-update curve, not ROC-AUC. The reported endpoint accuracies for both selected candidates were 74.0%, on validation.

This is online actor-recipe tuning, distinct from the offline Ridge epsilon sweep. We did not select using final test scores or expand the grid after seeing those final comparisons. It does not establish that all future cumulative Ridge critics should use large epsilon; the chosen .01 default belongs to a different matrix scale and recipe.

## Evidence and scope

Cluster: `outputs/transfer-ppo-v1/selection.json`, supplied in the project discussion. Pilot GPU costs are additional to main-run means. This selection is not a performance benchmark.
