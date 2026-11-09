# Can frozen compressed features support PPO with cumulative linear critics?

**The short pilot supported the integration: sparse Ridge reached 70.95% actor accuracy versus 68.92% for the tested PPO baseline. With only three seeds and 60 updates, it did not settle method rankings.**

| Method | Actor accuracy | Training GPUh/run |
|---|---|---|
| lstd-plain | 70.67% | 5.205 |
| lstd-sparse | 70.25% | 5.322 |
| lstd-two_thirds | 70.42% | 4.888 |
| ppo | 68.92% | 6.017 |
| ridge-sparse | 70.95% | 5.178 |

Historical two-fold, token-level study; actor seeds 201–203; 60 × 64 training answers; 2,000 greedy test questions; base accuracy 66.70%; LSTD lambda .99. Frozen features and cumulative raw matrix sums were used. Plain/sparse LSTD means were 70.67%/70.25%, which does not establish a sparsity benefit.

The pilot motivated the larger six-method comparison rather than a general claim about Ridge versus LSTD or PPO.

## Evidence and scope

[Recorded pilot summary](data/focused-pilot.json). Cluster: `outputs/numina-online-768-focused-v1/summary.json`. Reused validation calibration and inspected questions; not an independent benchmark.
