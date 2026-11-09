# Does an MLP readout improve offline predictions before extensive tuning?

**The frozen-SLICE MLP reached Brier 0.181589 versus raw Ridge’s 0.183459, but clipped Ridge was already 0.181007. Joint SLICE + MLP was weaker at 0.183017 in this initial three-seed pilot.**

| Offline predictor | Brier | Accuracy |
|---|---|---|
| Raw Ridge | 0.183459 | 73.162% |
| Clipped Ridge control | 0.181007 | 73.162% |
| Frozen SLICE + MLP | 0.181589 | 73.194% |
| Trainable SLICE + MLP | 0.183017 | 73.007% |

Three head seeds; 125,920 test prefixes from 2,000 questions. MLP outputs were sigmoid-bounded; Ridge was unbounded. Validation selected epochs. Comparing only against raw Ridge conflates nonlinear learning with bounded prediction. Clipping is a diagnostic control here, not a change to online Ridge.

The initial joint fit did not establish a benefit, motivating a separate tuned study. Reused inspected questions and previously selected compressor prevent an independent benchmark claim.

## Evidence and scope

[Pilot report extract](data/mlp-pilot-report.txt). Cluster: `outputs/slice-mlp-v1/report.txt`. Offline prediction only; no actor updates.
