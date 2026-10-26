# SLICE Critic

An internal Python library for compact value prediction on frozen LLM features.
Version **0.2.0** supports Qwen2.5-Math-1.5B and numeric math completions. It is not
published to PyPI or hosted as an API. It has no VERL or Slurm dependency.

The library generates answers, verifies correctness, extracts selected hidden
layers, trains a sparse/plain ReLU compressor, fits Ridge or LSTD, and publishes
an immutable critic artifact. A saved critic predicts eventual success for a
question plus partial answer; it does not certify reasoning or proofs.

## Install and train

```bash
python -m pip install -e './packages/slice-critic[train,dev]'
```

Keep pinned symbolic-verifier dependencies in a separate interpreter. On Duke,
use `scripts/install_critic_package.sh`; do not replace the container Torch.

```python
from slice_critic import TrainingConfig, train_critic

config = TrainingConfig(
    dataset="numeric-math.jsonl",       # Or a pinned-compatible HF dataset.
    question_column="problem",
    reference_column="answer",         # A single decimal/integer, not a solution.
    model_path="models/qwen-math",
    verifier_python=".venv-ppo-verifier/bin/python",
    max_questions=1000, responses_per_question=4, max_new_tokens=1024,
    layer_base=2, num_slices=5, compression_width=768,
    compression="sparse", method="ridge",
)
artifact = train_critic(config, output_dir="critics/math-v001")
```

Preview the resource limits before training. Numeric reference/column validation,
pinned dataset revisions, question-family splits, stage caches and checksums are
part of the recipe. Both outcome classes are required in train/validation.

## Load and score

```python
from slice_critic import Critic, State, inspect_artifact

metadata = inspect_artifact(artifact)   # CPU validation; no model download.
critic = Critic.load(artifact, model_path="models/qwen-math", device="cuda")
states = [State(question="What is 9 divided by 3?", prefix="Divide 9 by")]
features = critic.encode_batch(states)  # Detached CPU [states, width].
predictions = critic.score_batch(states)
```

The model is loaded once. Raw values and validation-calibrated probabilities are
separate outputs. Use exact `prefix_token_ids` for rollout alignment. Inputs are
never silently truncated. Serialize calls on one instance: extraction uses hooks;
padded GPU batching and attention-cache reuse are not implemented.

## Fit fresh online heads

```python
from slice_critic import CumulativeCritic

head = CumulativeCritic(all_question_ids, method="ridge", epsilon=.01)
values, weights, diagnostics = head.update(
    features_per_answer, terminal_rewards, response_masks, question_ids, step=1,
)
# The consuming project computes GAE and updates its actor.
```

For LSTD, specify `method="lstd", trace_lambda=.9` (or the desired tested trace).
Gamma is 1. Statistics accumulate in two persistent question folds; no head fits
its predicted question's rewards. Features stay frozen. Historical-policy data
are uncorrected. LSTD requires consecutive states, not isolated random prefixes.

Export only features to PPO; the offline value head is not an online baseline.
Exclude representation-training and selection families before collecting RL data.

```bash
python -m slice_critic preview --config config.json
python -m slice_critic train --config config.json --out critics/math-v001
python -m slice_critic inspect critics/math-v001/artifact
python -m slice_critic score critics/math-v001/artifact --states states.jsonl
python -m slice_critic export-filter critics/math-v001/artifact --out critics/math-v001/ppo-filter.pt
```

## Acceptance and distribution

```bash
python scripts/check_slice_release.py --out outputs/slice-release-check-v1
```

This builds and installs a wheel outside the checkout, blocks imports from the
research project and runs the package CPU tests. It keeps the checked wheel,
SHA256, dependency versions, log and verification report. Dependencies must
already be installed; no models are downloaded or packages published. GitHub CI
runs CPU acceptance; the Duke GPU smoke is separate.

Install that checked wheel in another project's prepared environment and provide
its artifact and matching model separately. Completed `slice-critic-v1` artifacts
remain supported. [API contract](API.md) · [changelog](CHANGELOG.md) ·
[project setup](../../SETUP.MD).
