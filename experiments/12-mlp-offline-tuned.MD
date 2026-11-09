# Does validation-tuned joint SLICE–MLP training clearly beat frozen Ridge?

**Joint training reached test Brier 0.180671 versus raw Ridge 0.183459 and clipped Ridge 0.181007. Its paired question-mean improvement over clipped Ridge had interval [−0.001019, +0.000453], so a clear gain beyond the bounded Ridge control was not established.**

| Locked test predictor | Brier | Prefix classification accuracy |
|---|---|---|
| Raw Ridge | 0.183459 | 73.162% |
| Clipped Ridge control | 0.181007 | 73.162% |
| Frozen SLICE + MLP | 0.181095 | 73.266% |
| Trainable SLICE + MLP | 0.180671 | 73.333% |

Three seeds 42–44; 125,920 test prefixes; 2,000 questions. Recipes were selected using seed-42 validation Brier, then all models were locked before test evaluation. Joint training started from each paired best frozen-head checkpoint; epoch zero was eligible. Selected head learning rate was 3e−4; joint head/compressor rates were 3e−5/3e−6 with activation penalty zero.

Question-mean Brier differences use a different weighting from the headline prefix-weighted table. Joint minus frozen was −0.000431, exploratory interval [−0.000821, 0]. The bootstrap was unadjusted and included the pilot seed. An offline prediction improvement does not establish improved PPO; that required the separate online study.

## Evidence and scope

Cluster: `outputs/slice-mlp-tuned-v1/validation-report.txt` and `report.txt`, supplied in the project discussion. No new RL training in this experiment.
