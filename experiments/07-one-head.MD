# Does removing question-fold cross-fitting hurt our linear PPO critics?

**In the reported five-seed comparison, one-head SLICE + Ridge reached 75.16%, up 0.94 pp over its two-fold replay comparator. LSTD changes were near zero; this motivated one shared cumulative head, without proving same-sample fitting is unbiased.**

| Linear PPO arm | One-head accuracy | Change vs paired two-fold replay |
|---|---|---|
| SLICE + Ridge | 75.16% | +0.94 pp |
| Raw two-thirds + Ridge | 72.84% | +0.23 pp |
| SLICE + LSTD(.9) | 72.16% | +0.06 pp |
| Raw two-thirds + LSTD(.9) | 72.36% | −0.04 pp |

Five actor seeds 301–305; 120 updates; 64 answers/update; token-level features and 2,000 final test questions. All four linear arms had completed five runs in the supplied snapshot. PPO had only four completed seeds then, so its partial mean is deliberately not used here.

One head uses all collected trajectories to fit and predict. Two-fold cross-fitting partitions fitting by question and predicts with the other fold. Removing it increases data available to each head but reintroduces same-sample reward dependence. Neither choice corrects historical-policy drift in cumulative statistics. The +0.94 pp change was positive in all five reported SLICE Ridge pairs, but no adjusted interval was supplied for this contrast.

## Evidence and scope

Cluster: `outputs/numina-no-folds-v1/*/result.json`, compared with `outputs/numina-prod-replay-v1`. Values transcribed from the completed-linear-run table supplied in the project discussion; the full no-fold summary is not stored in this checkout.
