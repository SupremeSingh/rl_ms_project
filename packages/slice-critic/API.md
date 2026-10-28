# SLICE public contract (0.2.x)

This is an internal alpha package. Import the symbols below from `slice_critic`;
implementation helpers may change. Version 1 supports the frozen
Qwen2.5-Math-1.5B and numeric final-answer tasks, not arbitrary architectures.

| Public symbol | Contract |
|---|---|
| `TrainingConfig` | Explicit data columns, geometric blocks, budgets, compressor and head; `validate()` rejects unsupported settings; `preview()` needs no download/GPU. |
| `train_critic(config, output_dir, device=...)` | Synchronous training; returns an immutable artifact path. Identical runs recover completed stages; changed code/config/model/runtime requires a new directory. |
| `inspect_artifact(path)` | Checks schema, hashes, tensor dimensions/finite values, layers and calibration on CPU; returns manifest, calibration, evaluation and `validated=True`. Never loads the encoder. |
| `State(question, prefix='', prefix_token_ids=None)` | A pre-action state. Supply prefix text or exact token IDs, not both; empty prefix means question-only. |
| `Critic.load(path, device=..., model_path=...)` | Validates before loading one persistent matching encoder. Relocation requires identical model/tokenizer hashes. |
| `critic.encode_batch(states)` | Ordered detached CPU float32 `[states, width]`; empty input gives `[0, width]`. No terminal value is appended. |
| `critic.score_batch(states)` | Ordered dictionaries with raw `value`, calibrated `probability`, `critic_version`, and `model_id`. Raw values remain unclipped. |
| `export_ppo_features(artifact, output)` | Checksummed feature-only checkpoint for the 768-wide PPO adapter. Rejects overwriting; excludes the offline value head. |
| `load_ppo_features(path, checksum, layers=..., eps=..., model_path=...)` | Checks checkpoint and supplied identities; returns frozen compressor, CPU mean and scale. Pass all compatibility arguments. |
| `encode_hidden_rollout_features(encoder, ids, masks, response_width, layer)` | Frozen raw-layer control readout; same causal alignment, no compressor. |
| `encode_rollout_features(encoder, ids, masks, response_width, fitted, layers, eps)` | Per-answer detached arrays for every pre-action response state. Consumer supplies the matching encoder, tokenizer and completion prompt. |
| `CumulativeCritic(question_ids, seed=..., epsilon=.01, trace_lambda=.99, method=...)` | Persistent question-fold matrix fitting. Explicitly specify `method` and `trace_lambda`; the compatibility constructor's trace default is zero. |

## Online fitting is separate from offline prediction

```python
from slice_critic import CumulativeCritic

critic = CumulativeCritic(all_training_question_ids, seed=42,
                         method="lstd", trace_lambda=.99, epsilon=.01)
for step, batch in enumerate(fresh_rollouts, start=1):
    values, heads, diagnostics = critic.update(
        batch.features, batch.terminal_rewards, batch.response_mask,
        batch.question_ids, step=step,
    )
    # The consuming RL project computes GAE and performs PPO actor updates.
```

Features are a list of `[answer_tokens, width]` arrays. Rewards/masks are padded
`[answers, max_answer_tokens]` tensors with contiguous nonempty masks. Gamma is
fixed to 1; only binary rewards at the final response token are supported.
Termination uses a zero successor, including its intercept. Consumers must define
truncation semantics for other tasks; our math project defines a capped task.

Two question folds persist: values for one fold use the other fold's matrices.
Ridge accumulates `phi.T @ phi` and `phi.T @ observed_return`; LSTD accumulates
trace-weighted TD products. Sums are never divided by data count. Epsilon is added
once at solve time, including on the intercept. Traces reset between trajectories.
Neither matrices nor heads backpropagate into the encoder.

Steps must be consecutive. `state_dict()` records statistics, not an implemented
actor-resume contract. The PPO runner requires new run directories after interrupted
actor training. Historical-policy statistics are uncorrected; this is not
guaranteed on-policy evaluation.

Exclude representation-training and validation-selection families from online data.
Question cross-fitting cannot undo supervised compressor leakage. A changed encoder
or compressor requires resetting/rebuilding matrices; do not mix feature bases.

## Compatibility and limits

Completed `slice-critic-v1` artifacts remain readable on this branch,
including the 0.1.0 cluster smoke artifacts. A code change invalidates unfinished
training recovery by design. Checksums support reproducibility, not authentication.

The pinned Duke container is the reference GPU runtime. CPU acceptance uses a toy
causal backend and does not establish real-model accuracy or throughput. Scoring
currently processes states individually; serialize calls on one instance because
extraction uses temporary hooks. There is no remote registry, service, access
control or padded inference optimization.
