# SLICE Critic

A standalone Python package for frozen multi-layer LLM features, learned compact
bottlenecks and linear correctness predictors. It has no dependency on VERL,
this research repository, a scheduler or J-lens.

Version 1 supports the frozen `Qwen/Qwen2.5-Math-1.5B` encoder and text questions
with single decimal/integer reference answers. Compatible Hugging Face datasets
and local JSONL files can be used; arbitrary task types are not supported.

## Install

From the research checkout, install the package into a suitable Torch environment:

```bash
python -m pip install -e './packages/slice-critic[train]'
```

The package is not published on PyPI. To use it in another project, build a wheel
with `python -m pip wheel --no-deps ./packages/slice-critic -w dist` and install
that wheel with its `train` extra. Base dependencies are Torch and NumPy;
Transformers, datasets and Hugging Face Hub are optional training dependencies.
There is no automatic Slurm submission, API service or billing layer.

Symbolic verification should use a separate interpreter because its pinned SymPy
dependencies can conflict with Torch. Install the exact symbolic dependencies listed in the `verifier` extra of
`pyproject.toml`, then install this wheel with `--no-deps` to keep that interpreter
Torch-free. A normal extras install also resolves the base Torch dependency.
The verifier entry point does not import Torch. On Duke, use the existing
`.venv-ppo-verifier` environment and the repository's install commands below.

## Train

```python
from slice_critic import TrainingConfig, train_critic

config = TrainingConfig(
    dataset="organization/numeric-math",  # Or a local JSONL path.
    question_column="problem",
    reference_column="answer",           # A single integer/decimal, not reasoning.
    solution_column=None,                 # Optional boxed reference-solution audit.
    max_questions=1000,
    responses_per_question=4,
    max_new_tokens=1024,
    layer_base=2,
    num_slices=5,
    compression_width=768,
    compression="sparse",                # Or "plain".
    method="ridge",                      # Or "lstd", gamma=1, critic_lambda=.99.
    verifier_python="/path/to/verifier/bin/python",
)
print(config.preview())                  # No model loading or network access.
artifact = train_critic(config, output_dir="critics/math-v001", device="cuda")
```

For 28 blocks, base 2/count 5 resolves to **28, 27, 25, 21, 13** (one-based).
Distances are floor(base**i); ties advance to the next unused distance. Base 1
therefore selects consecutive final blocks. Out-of-range spreads are rejected.
Saved block indices, not recalculated defaults, govern artifact loading.

Only the fixed Qwen math encoder is supported in version 1. A local `model_path`
can reuse existing weights; otherwise the model and Hugging Face dataset are
downloaded at resolved commit revisions. We enforce the native 4096-token context.
Explicit question/reference column names support datasets with different schemas.
References containing fractions, symbolic expressions, sets or worked solutions
are rejected. Do not point `reference_column` at GSM8K's full reasoning field;
first provide a normalized numeric answer column.

Questions are grouped by number-template family before a seeded 60/20/20 split.
Selection never uses model success. The requested number of eligible families
must be available; the recipe does not silently relax filters. This grouping is
not semantic or pretraining decontamination. With `solution_column`, require one
numeric reference box and consistency with the saved answer.

The frozen actor generates complete answers. Math-Verify labels them; statuses
are saved. Only nonempty answers scored `correct` or `incorrect` enter fitting.
Sample prefix lengths uniformly without replacement from **0,...,T-1**; take at
most `prefixes_per_answer=8` per answer. Extract the hidden state immediately
before the next action; future tokens cannot enter that feature. Raw intermediate
representations and generated token IDs are retained in the training directory.

Train the compressor on training prefixes using BCE; sparse adds 0.01 times mean
ReLU activation. Validation Brier chooses its epoch at one configured learning
rate. Freeze it, then fit the linear head on **all retained training transitions**.
Ridge uses complete outcomes; LSTD uses adjacent features and episode-reset traces.
Both use raw accumulated sums and epsilon=.01, including the intercept. The sampled feature splits are held in CPU memory, so budget RAM as well as GPU
time when increasing N. This is frozen-policy supervised/value prediction, not an
actor update or online PPO.

Train and validation must each contain both outcome classes. Calibrate raw head
values on validation only. Test evaluation occurs after model choices are fixed;
report Brier, correctness-prediction accuracy, a constant baseline and exclusion
counts. Prefixes within a question are correlated; reported metrics are not an
independent-sample significance test or a math-solving accuracy benchmark.

## Load and score

```python
from slice_critic import Critic, State

critic = Critic.load("critics/math-v001/artifact", device="cuda")
states = [State(question="What is 9 divided by 3?", prefix="Divide 9 by")]
features = critic.encode_batch(states)    # Detached CPU tensor: [states, width].
results = critic.score_batch(states)      # Raw value, calibrated probability, version.
```

The encoder is loaded once. The batch interface preserves order and currently
encodes states individually; it does not yet optimize padded GPU batching or
reuse attention caches. Serialize calls to one scorer instance because extraction
uses temporary model hooks. The compressor and linear dot product run on CPU.
For exact rollout alignment, pass `prefix_token_ids` instead of prefix text.
Input context is never silently truncated. Artifact checksums and exact local
encoder/tokenizer file hashes must match; use `model_path=` to relocate the same
model snapshot. Raw values are unclipped; calibrated probabilities are separate.

A value estimates terminal correctness under the training continuation policy.
It is not a correctness certificate or a guarantee under a changed search policy.

## CLI and artifacts

Put the TrainingConfig fields in `config.json`:

```bash
slice-critic preview --config config.json
slice-critic train --config config.json --out critics/math-v001
slice-critic inspect critics/math-v001/artifact
slice-critic score critics/math-v001/artifact --states states.jsonl
```

Each states.jsonl row has `question` and optional `prefix` or `prefix_token_ids`.
The artifact contains `manifest.json`, `checkpoint.pt`, `calibration.json` and
`evaluation.json`. Publish versions are immutable. It records selected blocks,
model hashes, pinned data revision, question families, prompt, generation settings,
implementation hashes and verification provenance. The training directory also
contains question snapshots, generated answers, feature caches and head statistics.

Re-running the same training command reuses completed stage records after identity
checks. A local writer lock prevents concurrent updates. Head checkpoints mark
whole processed trajectories so interruption does not double-count data. Changed
code, configuration, model or cached checksums fail clearly. This is stage/trajectory
recovery; interrupted compressor optimization restarts until its checkpoint exists.
Preserve a damaged checkpoint for inspection rather than bypassing integrity checks.

## Reuse in PPO

```python
from slice_critic import CumulativeCritic

head = CumulativeCritic(question_ids, seed=42, method="ridge", epsilon=.01)
values, heads, diagnostics = head.update(
    features, terminal_rewards, response_mask, batch_question_ids, step=1
)
```

This is the same persistent question-cross-fitted implementation used by our PPO
project. `features` is a list of per-answer [tokens,width] arrays; rewards and
masks are padded [answers,tokens] tensors. The encoder remains frozen, matrices
persist, and actor updates use fresh trajectories. Historical-policy data are
uncorrected. For LSTD, pass `method="lstd", trace_lambda=.99` explicitly. LSTD requires
consecutive transitions, not just sampled prefixes.

For the existing 768-wide PPO worker configuration:

```bash
slice-critic export-filter critics/math-v001/artifact --out critics/math-v001/ppo-filter.pt
```

Export only the frozen feature basis; the static offline value weights are not
used for PPO advantages. Exclude representation-training and validation-selection
question families from subsequent RL data. Head cross-fitting cannot undo leakage
through supervised feature training. The package records these families for the
consumer to enforce. The original Numina comparison launcher still requires its
original offline study protocol; exporting a new filter does not make an arbitrary
dataset artifact interchangeable with that locked experiment.
