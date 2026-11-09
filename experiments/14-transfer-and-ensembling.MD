# Does transferring a full critic beat refitting its head or combining multiple SLICEs?

**On GSM8K, Numina SLICE + refitted Ridge improved question-mean Brier to 0.204976 from 0.216804 for the transferred full critic. Joint-data SLICE scored 0.206014 and the 768-feature combined system 0.223276; the RMS two-thirds control was best at 0.201330.**

| Offline arm | Question-mean raw Brier |
|---|---|
| Full Numina critic transfer | 0.216804 |
| Numina SLICE + target Ridge | 0.204976 |
| Target-only SLICE + target Ridge | 0.216722 |
| Pooled Numina/GSM8K SLICE | 0.206014 |
| Combined 768-feature representation | 0.223276 |
| Wide pooled control | 0.209069 |
| RMS two-thirds + target Ridge | 0.201330 |

Three compressor seeds 42–44; 1,000 eligible GSM8K test questions and 30,849 retained prefixes. The source/target collections each used 2,000 train / 500 validation / 1,000 test questions with four generated answers/question. Source means on its own Numina test were approximately 0.191740 question-mean Brier, but cross-dataset difficulty and outcome prevalence prevent interpreting the difference as pure transfer degradation.

Full transfer retains source features and head. Feature transfer freezes the source compressor and refits Ridge on target labels. Pooled training learns one compressor from both datasets under matched question budgets. The combined 768 arm concatenates two 768-feature specialists and learns a linear 1,536→768 projection plus temporary supervision head; it is not justified to average unaligned feature coordinates directly. A larger wide pooled control checks whether width alone explains behavior; the legacy `tools/transfer/study.py` at commit `0a62a4e` defines this combiner.

Full minus feature transfer Brier difference was +0.011828, exploratory interval [0.007188, 0.017012]. Combined minus pooled was +0.017262, interval [0.011612, 0.022063]. These results do not support an ensemble advantage. The raw layer itself never learns or transfers; its target Ridge head is trained anew.

## Evidence and scope

[Recorded seven-arm summary](data/transfer-seven-arms.json). Cluster: `outputs/slice-transfer-numina-gsm8k-v2/summary.json`. Numeric eligible subsets; retained verifiable answers; three seeds; six unadjusted exploratory contrasts; no PPO or cost superiority claim.
