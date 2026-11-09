# Do transferred SLICE features improve PPO over a normalized single layer?

**The five-seed means were close: transferred SLICE reached 74.01% actor accuracy versus 73.74% for the RMS layer, with adjusted difference interval [−4.35, 4.22] pp. SLICE used 6.227 versus 6.012 training GPUh, so this study established neither higher accuracy nor cheaper online training.**

| Transferred-feature PPO arm | Final actor accuracy | Accuracy-curve AUC | Training GPUh/run | Accounted GPUh/run |
|---|---|---|---|---|
| SLICE + fresh Ridge | 74.01% | 0.6926 | 6.227 | 6.521 |
| RMS two-thirds + fresh Ridge | 73.74% | 0.6891 | 6.012 | 6.326 |

Five fresh actor seeds (701–705); 60 updates × 64 answers; new cumulative Ridge heads, not transferred offline head weights. Penalties locked by the separate validation pilot: SLICE 10,000; layer 1,000. The target is the eligible MATH numeric subset, using the clarification study’s data; this is not a fresh Numina comparison against GRPO or conventional PPO. Offline source data used a 1,024-token cap, while PPO used 2,048.

The report’s unrounded final-accuracy gain was +0.2721 pp. Its descriptive interval was [−3.54, 3.81] pp and adjusted interval [−4.35, 4.22] pp. Accuracy-curve AUC gain was +0.3515 pp, adjusted interval [−1.49, 2.30] pp. AUC measures learning-curve area, not ROC discrimination. Intervals spanning zero do not demonstrate equivalence.

The compact representation transfers usefully, but fewer head coordinates did not create a measured end-to-end speedup. Prior compressor training, tuning and any failed attempts are additional costs. The actor continues to receive fresh data; historical-policy critic statistics remain uncorrected.

## Evidence and scope

Cluster: `outputs/transfer-ppo-v1/report.txt`, supplied in the project discussion. The historical implementation (`tools/transfer/online_transfer.py` at commit `9f39e73`) explicitly selects domain `math`; the collection audit retained 294 test questions, rather than the complete MATH500 benchmark. See [cost accounting](16-cost-accounting.MD) for scope distinctions.
