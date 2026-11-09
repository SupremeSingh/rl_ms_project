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

The current recipe uses one shared cumulative system: all training questions
contribute to the same matrices, and the fitted head predicts their values.
Historical two-fold studies instead assigned questions permanently to two systems
and used opposite-fold predictions. Those results are labeled separately in the
[experiment records](../experiments/07-one-head.MD). Supervised
representation-fitting families are excluded from RL data. Neither cross-fitting
nor a fixed feature basis corrects historical-policy drift. Actor updates always
use fresh rollouts.

## Current cumulative algorithm

The accuracy-focused recipe uses token-level states, epsilon .01, gamma 1 and
GAE lambda .95. LSTD's eligibility-trace lambda is a separate setting. An
intercept makes the 768-feature SLICE head 769-dimensional.

```text
Freeze the base encoder, learned bottleneck and feature normalization.
Initialize one A = 0 and b = 0.

For each PPO update:
    Generate fresh answers with the current actor; verify final outcomes.
    Extract frozen features at each pre-action prefix; append intercept 1.

    For each answer:
        Ridge:
            For each prefix x:
                A += outer(x, x)
                b += x * final_reward

        LSTD:
            Reset eligibility trace z = 0.
            For each transition (x, reward, x_next):
                z = gamma * critic_lambda * z + x
                A += outer(z, x - gamma * x_next)
                b += z * reward
            # Reward is zero except at the last action.
            # Terminal x_next is all zero, including its intercept.

    w = solve(A + epsilon * I, b)
    Predict values with this shared head: value(x) = dot(w, x).
    Compute GAE from these values, fresh rewards and terminal value zero.
    Update the actor with PPO on fresh answers only.

    Retain raw A and b; do not divide by the sample count.
    Keep encoder, bottleneck and normalization frozen.
```

Chunk experiments use chunk-start states and chunk transitions instead, then
broadcast each chunk advantage to its original action tokens. GAE lambda .95
therefore decays per chunk rather than per token. This also changes state
weighting and the relative strength of epsilon; it is a separate recipe, not
just a faster implementation of token-level learning. See the
[matched comparison](../experiments/19-token-versus-chunk.MD).

Conventional PPO retains its own trainable transformer critic. GRPO retains
within-question group normalization. These are baselines with the same prompt,
reward, rollout and update budgets, not alternative implementations of SLICE.

The two-message parquet carrier from the completed study is accepted for loading
compatibility. It never changes the model prompt: all paths encode the same plain
completion text from `slice_critic.config.completion_prompt`.
