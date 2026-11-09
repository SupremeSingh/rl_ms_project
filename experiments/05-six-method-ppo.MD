# Does sparse Ridge PPO compete with GRPO and a trainable transformer critic?

**Sparse Ridge had the highest mean actor accuracy, 74.60%, with 10.346 training GPUh versus PPO’s 73.52% and 12.513 GPUh (17.3% less online training compute). Its adjusted +1.08 pp interval against PPO crossed zero, so superiority over PPO was not established.**

| Method | Actor accuracy | Training GPUh/run |
|---|---|---|
| grpo | 72.75% | 9.375 |
| lstd-sparse | 70.86% | 10.329 |
| lstd-two_thirds | 72.40% | 9.613 |
| ppo | 73.52% | 12.513 |
| ridge-sparse | 74.60% | 10.346 |
| ridge-two_thirds | 72.54% | 10.154 |

Historical two-fold token-level protocol; five actor seeds 301–305; 120 × 64 training answers; 2,000 greedy test questions; 2,048-token cap; two A5000 GPUs/run. LSTD lambda .9; GAE lambda .95; gamma 1; epsilon .01. Base accuracy was 66.70%.

These studies are historical comparisons, not the present one-head recipe. The replay reused questions, actor seed IDs, compressor and PPO profile, but started fresh actors and heads. Bootstrap intervals are approximate and adjusted across eight primary contrasts. Calibration and offline costs are additional. The original had one PPO target-MSE health failure; the replay passed the checks, which do not prove held-out value accuracy.
Original adjusted intervals: sparse Ridge minus two-thirds Ridge [0.59, 3.53] pp; minus PPO [−1.22, 3.56] pp; minus GRPO [0.08, 3.50] pp.

## Evidence and scope

[Machine-readable results](../docs/results/numina-six-method.json) and [preserved interpretation](data/previous-readme-results.md). See [current setup](../SETUP.MD) for environment requirements; do not treat historical manifests as current defaults.
