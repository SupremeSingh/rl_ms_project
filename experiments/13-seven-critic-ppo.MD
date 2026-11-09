# Do neural heads or low-lambda LSTD improve PPO over cumulative Ridge?

**Ridge led the seven-arm means at 73.97%, versus 72.23% for frozen-SLICE MLP and 72.75% for joint SLICE–MLP. LSTD(.99) was close at 73.62%; LSTD(.05) fell to 70.22%, so neither a neural head nor low lambda improved this recipe.**

| Method | Actor accuracy | Training GPUh/run |
|---|---|---|
| SLICE Ridge | 73.97% | 10.367 |
| Frozen SLICE + MLP | 72.23% | 10.417 |
| Joint SLICE + MLP | 72.75% | 10.677 |
| LSTD(.05) | 70.22% | 10.111 |
| LSTD(.99) | 73.62% | 10.609 |
| Conventional PPO | 72.65% | 12.529 |
| GRPO | 70.73% | 8.766 |

Three new actor seeds 501–503; 120 × 64 answers; same inspected Numina data and selected compressor; token-level GAE lambda .95, gamma 1, epsilon .01. One cumulative system for Ridge/LSTD. Neural critics used fresh sigmoid heads and fresh GAE-return fitting; no offline MLP head initialization or old-coordinate matrix accumulation for trainable features. Neural fitting ran on one owning actor-worker GPU; all arms allocated two A5000 GPUs, and transport overhead was timed.

Adjusted intervals (percentage points): frozen MLP minus Ridge [−3.60, −0.005]; joint MLP minus Ridge [−4.20, 1.55]; LSTD(.05) minus Ridge [−8.44, −0.35]; LSTD(.99) minus Ridge [−2.20, 1.30]; Ridge minus PPO [−0.99, 3.82]; Ridge minus GRPO [0.40, 6.64]. Three seeds and approximate bootstrap limit precision; no universal architecture ranking follows.

Ridge used approximately 17.3% less online training compute than tested PPO, but more than GRPO. LSTD remains useful as a supported bootstrapped alternative; better conditioning and longer traces do not guarantee better actor learning. This does not establish a general rule that short horizons require Ridge and long horizons require LSTD.

## Evidence and scope

Cluster: `outputs/numina-seven-critics-gpu-v1/report.txt`, supplied in the project discussion. Fresh actors, not a new independent dataset. No claim that sigmoid clipping was applied to Ridge.
