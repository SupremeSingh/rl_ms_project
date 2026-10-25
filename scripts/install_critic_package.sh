#!/usr/bin/env bash
# Run inside the pinned container, in a CPU allocation; never during active jobs.
set -euo pipefail
[[ -f packages/slice-critic/pyproject.toml ]] || { echo 'Run from repository root.' >&2; exit 1; }
for environment in .venv .venv-critic .venv-ppo-verifier; do
  [[ -x "$environment/bin/python" ]] || { echo "Missing $environment; run its setup first." >&2; exit 1; }
done
# Reuse each environment's existing Torch/Transformers/verifier versions.
for environment in .venv .venv-critic .venv-ppo-verifier; do
  .venv/bin/python -m pip --python "$PWD/$environment/bin/python" \
    install --no-deps -e ./packages/slice-critic
done
for environment in .venv .venv-critic; do
  "$environment/bin/python" -c 'from slice_critic import TrainingConfig, CumulativeCritic, Critic; print(TrainingConfig(dataset="preview").preview()["blocks_one_based"])'
done
.venv-ppo-verifier/bin/python -c 'from slice_critic.grading import provenance; print(provenance()["packages"])'

# Record the actual environments before preparing any new experiment manifest.
.venv/bin/python -m pip freeze > configs/environment.txt
.venv/bin/python -m pip --python "$PWD/.venv-critic/bin/python" freeze > .venv-critic/environment.txt
.venv/bin/python -m pip --python "$PWD/.venv-ppo-verifier/bin/python" freeze > configs/ppo-verifier-environment.txt
