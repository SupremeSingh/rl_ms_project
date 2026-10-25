# Reusable critic package

The standalone package lives at [`packages/slice-critic`](../packages/slice-critic).
Its import name is `slice_critic`, distribution/CLI name is `slice-critic`, and
version is 0.1.0. It is not yet published to PyPI. The package can be copied to
another repository or installed from a built wheel. It never imports `math_rl`,
VERL, J-lens or cluster job-management code.

The [package README](../packages/slice-critic/README.md) documents the API, CLI,
column selection, geometric layer selection, artifact format, prediction semantics
and the limited numeric-math recipe. Arbitrary datasets and model architectures
are not claimed supported. Public Hugging Face datasets and local numeric-reference
JSONL are supported; generation and scoring require the matching frozen model.

## What moved

| Component | Package module | Existing project compatibility path |
|---|---|---|
| Sparse/plain compressor | `slice_critic.bottleneck` | `math_rl.sparse_bottleneck` |
| RMS/hidden feature transforms | `slice_critic.features` | `math_rl.critic_features` |
| Causal layer capture | `slice_critic.layers` | `math_rl.critic_layers` |
| PPO-compatible bottleneck readout | `slice_critic.online` | `math_rl.bottleneck_features` |
| Persistent cross-fitted Ridge/LSTD | `slice_critic.cumulative` | `math_rl.cumulative_lstd` |

The old paths re-export the same classes/functions so existing tensor checkpoints
and worker imports remain usable. Actor rollouts, Math-Verify reward integration,
GAE, distributed updates and Slurm orchestration remain in this project.
Provenance now also hashes the package sources. Completed studies remain intact;
new package code requires a fresh experiment directory rather than relaxing old
manifest checks.

## Install on the existing Duke environments

Publish/sync the code first, with no running jobs using this checkout. Enter a CPU
allocation before changing environments:

```bash
srun -p compsci --cpus-per-task=8 --mem=128G --time=00:45:00 --pty bash -i
source scripts/cluster_env.sh
bash scripts/container_exec.sh bash scripts/install_critic_package.sh
exit
```

This installs the same source package into `.venv`, `.venv-critic` and the isolated
`.venv-ppo-verifier`, without replacing their pinned dependencies. New environment
setup scripts also install the package. The verifier only uses the package's lazy,
Torch-free grading entry point.

Keep the verifier executable at `.venv-ppo-verifier/bin/python`: resolving its
symlink to system Python bypasses the virtual environment. Training now checks
the verifier imports and pinned versions before loading the model, and reports
child-process stderr if verification fails.

## Preview and train a new dataset

Create a JSON config using the fields in `TrainingConfig`. No model or dataset is
downloaded by preview:

```bash
bash scripts/container_exec.sh "$PWD/.venv-critic/bin/python" -m slice_critic \
  preview --config my-critic.json
```

Within an appropriate GPU allocation, run training:

```bash
bash scripts/container_exec.sh "$PWD/.venv-critic/bin/python" -m slice_critic \
  train --config my-critic.json --out outputs/my-critic-v1
```

Example `my-critic.json` (replace the dataset ID and columns):

```json
{
  "dataset": "organization/numeric-math",
  "question_column": "problem",
  "reference_column": "answer",
  "model_path": "models/qwen-math",
  "verifier_python": ".venv-ppo-verifier/bin/python",
  "max_questions": 1000,
  "responses_per_question": 4,
  "max_new_tokens": 1024,
  "layer_base": 2,
  "num_slices": 5,
  "compression_width": 768,
  "compression": "sparse",
  "method": "ridge"
}
```

There are no downloads or GPU jobs until you explicitly train. A dataset's numeric
answer column is required; worked solutions need preprocessing. HF revisions are
resolved to immutable commits, local JSONL is hashed, and the family splits are
saved before generation. Do not use success-based sampling to make the dataset
look balanced. The trainer fails clearly if retained train/validation examples
lack both classes.

## Package smoke test

After installation, from the login node:

```bash
mkdir -p logs
sbatch scripts/submit_slice_smoke.sh outputs/slice-package-smoke-v1
```

This uses 40 existing **training** questions to exercise the new standalone path:
generate new answers, verify, extract/store features, train a sparse-768 compressor,
fit Ridge, calibrate/evaluate, publish, load, score, and export a PPO-compatible
feature checkpoint. It requests one GPU with a four-hour limit and produces:

```bash
cat outputs/slice-package-smoke-v1/verification.json
```

Both outcome classes are required; too few retained successes/failures is reported
as a data limitation rather than silently weakening the checks. This tiny run is
an integration check, not a scientific accuracy result. It does not update an actor.
Existing six-method smoke tests cover the project PPO adapter, while local tests
check that the adapters use the package's unchanged fitting implementations.
The new real-model package smoke still needs to be run on the cluster.

## Package-to-PPO integration

After the standalone package smoke passes, use its published artifact in a fresh
two-GPU job. This runs two actual actor updates each for Ridge and LSTD(0.99):

```bash
source scripts/cluster_env.sh
mkdir -p logs
sbatch scripts/submit_slice_ppo_smoke.sh \
  outputs/slice-package-smoke-v2/training/artifact \
  outputs/slice-ppo-smoke-v1
```

Check `outputs/slice-ppo-smoke-v1/status.json` while it runs and
`outputs/slice-ppo-smoke-v1/verification.json` when complete. Method logs are
`ridge.log` and `lstd.log` inside that output directory. Allow up to the requested
four hours; actual time depends on the allocation and generation lengths.

The bridge checks the artifact and encoder identities and exports only the
compressor, normalization and layer metadata. It excludes representation-training
and selection question families, takes 64 online training and 16 validation
questions from the saved Numina split, and does not read final test scores. Each
method starts with empty statistics, then accumulates two fresh 64-answer batches
in two persistent question folds, with epsilon 0.01, gamma 1 and no buffer.
GAE uses lambda 0.95; LSTD uses lambda 0.99. The encoder and compressor stay frozen.
The offline value head and probability calibration are not used by PPO.

Verification checks real actor completion, finite cross-fitted heads and matrices,
769-dimensional systems including the intercept, and retention of both updates.
This checks integration, not scientific accuracy or convergence. Larger studies
should use a properly trained artifact rather than the 40-question smoke artifact.

## Distribution and ownership

This is currently an **internal, source-controlled package** in this repository,
installed in editable mode into the cluster environments. It is not published to
PyPI, hosted as a service, or automatically shared with other users. Another
project can install a built wheel and load a compatible artifact locally; give it
access to the matching model and pinned verifier environment as needed.

Code distribution and trained artifacts are separate. Artifacts are local,
checksummed directories with model/layer/configuration provenance. They have no
remote registry or access-control service. A future registry or public package
release is a separate deployment decision. The package owns frozen features and
linear fitting; this repository owns rollouts, GAE, PPO, VERL and Slurm jobs.

## Remaining scope

This extraction does not implement the proposed approval API, shared registry
server or remote scoring. Training directories and immutable artifacts are the
initial registry building blocks. `encode_batch` gives consumers the frozen
representation; `score_batch` uses the published static head. PPO must fit its own
changing head and exclude the recorded representation-training/selection families.
An offline trained head is not silently substituted for the cross-fitted PPO head.
