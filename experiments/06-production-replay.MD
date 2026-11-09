# Does the cleaned-up pipeline reproduce the earlier six-method pattern?

**Sparse Ridge again had the highest mean, 74.22%, versus 72.61% for two-thirds Ridge; the adjusted gain interval was [0.17, 3.19] pp. PPO’s mean changed to 69.93%, so the replay reproduced the main pattern rather than identical outcomes.**

| Method | Actor accuracy | Training GPUh/run |
|---|---|---|
| grpo | 72.42% | 8.681 |
| lstd-sparse | 72.10% | 10.244 |
| lstd-two_thirds | 72.40% | 9.774 |
| ppo | 69.93% | 12.166 |
| ridge-sparse | 74.22% | 10.394 |
| ridge-two_thirds | 72.61% | 9.768 |

Historical two-fold token-level protocol; five actor seeds 301–305; 120 × 64 training answers; 2,000 greedy test questions; 2,048-token cap; two A5000 GPUs/run. LSTD lambda .9; GAE lambda .95; gamma 1; epsilon .01. Base accuracy was 66.70%.

These studies are historical comparisons, not the present one-head recipe. The replay reused questions, actor seed IDs, compressor and PPO profile, but started fresh actors and heads. Bootstrap intervals are approximate and adjusted across eight primary contrasts. Calibration and offline costs are additional. The original had one PPO target-MSE health failure; the replay passed the checks, which do not prove held-out value accuracy.
Replay adjusted intervals: Ridge minus GRPO [−0.16, 3.80] pp; minus PPO [−2.28, 12.60] pp. Neither establishes superiority. PPO seed accuracies ranged from 61.95% to 75.80%.

## Evidence and scope

[Machine-readable results](../docs/results/numina-prod-replay.json) and [preserved interpretation](data/previous-readme-results.md). See [current setup](../SETUP.MD) for environment requirements; do not treat historical manifests as current defaults.
