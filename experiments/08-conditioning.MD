# Does poor matrix conditioning explain why LSTD trails Ridge?

**Not by itself: at update 120, seed 301’s regularized SLICE Ridge matrix had condition number 3.06×10⁴, while LSTD’s was better conditioned at 2.70×10³. Better conditioning did not guarantee better actor learning.**

| System, seed 301 | epsilon | Condition number | Smallest singular value |
|---|---|---|---|
| SLICE Ridge | 0.01 | 3.060603e4 | 4.093297e4 |
| SLICE LSTD(.9) | 0.01 | 2.702353e3 | 2.163175e4 |
| Raw two-thirds Ridge | 0.01 | 6.505e6 | 5.925e3 |

Read-only analysis of saved one-head matrices, not new PPO training. The condition number is σmax/σmin of A + epsilon I. Ridge A is a Gram matrix; LSTD A is generally nonsymmetric and depends on feature differences and eligibility traces. Their spectra need not be ordered as intuition about temporal correlation suggests.

A large condition number measures worst-case sensitivity, not actual prediction error. Target alignment, feature adequacy, policy drift and bootstrapping still matter. At the saved scale, increasing SLICE Ridge epsilon from .01 to 100 barely changed conditioning; absolute epsilon has meaning only relative to the accumulated matrix scale.

## Evidence and scope

[Twenty-system audit extract](data/conditioning-report.txt). Cluster: `outputs/conditioning-no-folds-v1/conditioning.json`. The precise seed-301 SLICE singular values were also supplied by the direct checkpoint inspection.
