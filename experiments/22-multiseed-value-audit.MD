# Which estimator best tracks its own actor's value across more seeds and questions?

**Not measured yet: this repeat uses three fresh actor seeds, 50 evaluation
questions and 128 Monte Carlo continuations per state. It preserves conventional
PPO's original training procedure and each method's own actor updates; it is
not the fixed-policy MC-regression experiment.**

The six arms and 120 × 64 actor-training budget match the first six-actor pilot.
Evaluation states are shared within each seed, question IDs match across seeds,
and each method's final actor supplies its own reference-value estimates.
Monte Carlo outcomes are used only for evaluation, never as substitute PPO
critic-training targets.

Results will include per-seed and combined scatter plots, mean MAE/RMSE, seed
variation, paired seed/question intervals and actor validation accuracy. A
matched-coverage progress plot uses sampled-prefix order rather than uneven
token-progress bins. Maximum reference budget: 460,800 completions, plus bank
trajectories and independent GRPO groups.

## Evidence and scope

[Protocol and commands](../docs/VALUE_REPEAT.md). Planned main output:
`outputs/numina-value-repeat-v1/main`. Same inspected Numina question pool and
compressor; three seeds and finite MC samples do not establish a universal
ranking or exact ground truth. No completed results yet.
