# Boundaries and invariants

`slice-critic` is a standalone library. It owns the completion-prompt contract,
causal layer readout, normalization, compressor, numeric/reference verification,
Ridge/LSTD equations, artifact validation and persistent scoring. It imports no
research project, VERL or scheduler code. Install a checked wheel to reuse it.

`math_rl` consumes the public feature and cumulative-head interfaces. It owns
Numina selection, actor rollouts, the VERL GAE/PPO/GRPO integration, baseline
health checks, experiment locking and paired reporting. Shell scripts own only
environment setup and scheduler submission. The offline and online value heads
serve different policies; the static fitted head is never passed to PPO.

| Module | Responsibility |
|---|---|
| `slice_critic.training` / `backend` | Frozen-policy data generation and supervised representation training. |
| `slice_critic.online` / `layers` | Detached pre-action features, shared by online consumers. |
| `slice_critic.cumulative` / `linear` | Raw-sum matrices, question folds, traces and solves. |
| `slice_critic.artifacts` / `storage` / `bridge` | Validated artifacts, atomic files, PPO feature export. |
| `math_rl.data` / `reproduction` | Pinned data preparation and portable final-study input capture. |
| `math_rl.experiment` / `runner` | One six-method recipe, validation-only selection and reporting. |
| `math_rl.main` / `trainer` / `workers` | Pinned distributed actor training. |
| `math_rl.ppo_reward` / `verifier_batch` | JSON transport to the package's isolated verifier. |

A critic state is the prompt plus answer tokens strictly before an action. The
terminal reward belongs at the final sampled token. LSTD's terminal successor is
zero, including its intercept. Gamma is 1; traces reset between answers. Matrix
sums persist without per-update normalization. The encoder/compressor must not
change while those sums are retained.

In the original/default two-fold mode, question folds remain fixed across all
historical and current answers; values use the opposite fold's matrices.
The v2 experiment also supports `critic_folds=1`: one system fits all historical
and current answers before predicting the current batch, without cross-fitting.
Both modes keep the feature basis fixed and accumulate raw sums.
Supervised representation-fitting
families are excluded from RL data; cross-fitting alone cannot undo that leakage.
Actor updates always use fresh rollouts, even when critic statistics include
uncorrected older-policy data.

The no-fold six-method study and exact commands are in [NO_FOLDS.md](NO_FOLDS.md).

Conventional PPO retains its own trainable transformer critic. GRPO retains
within-question group normalization. These are baselines with the same prompt,
reward, rollout and update budgets, not alternative implementations of SLICE.

The two-message parquet carrier from the completed study is accepted for loading
compatibility. It never changes the model prompt: all paths encode the same plain
completion text from `slice_critic.config.completion_prompt`.
