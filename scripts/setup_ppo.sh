#!/usr/bin/env bash
# Run inside the pinned container, in a compute allocation.
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Run setup in a compute allocation.' >&2; exit 1; }
python3 -m venv --without-pip .venv-ppo-verifier
PIP_EXTRA_INDEX_URL='' .venv/bin/python -m pip --python .venv-ppo-verifier/bin/python \
  install --index-url https://pypi.org/simple -r requirements-verifier.txt
PYTHONPATH=src .venv-ppo-verifier/bin/python -c \
  'from math_rl.math_verify_reward import provenance; print(provenance())'
.venv/bin/python -m pip --python .venv-ppo-verifier/bin/python freeze > configs/ppo-verifier-environment.txt
MATH_RL_TEST_VERIFIER="$PWD/.venv-ppo-verifier/bin/python" \
  .venv/bin/python -m pytest -q tests/test_ppo.py tests/test_grpo.py
.venv/bin/python -m math_rl.train --config-name ppo --cfg job --resolve
.venv/bin/python -m math_rl.train --config-name grpo --cfg job --resolve
