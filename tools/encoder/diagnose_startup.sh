#!/usr/bin/env bash
# Bounded, CPU-only diagnostics; no deletes, installs, model downloads or training.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/cluster_env.sh
export MATH_RL_GPU=0 MATH_RL_STARTUP_TRACE=1
echo '=== Checkout and active jobs ==='
git branch --show-current
git rev-parse HEAD
git status --short
squeue -u "$USER" -o '%.18i %.12j %.10T %.30R' || true
echo '=== Environment and storage ==='
ls -ld .venv .venv-critic .venv-encoder || true
df -h . "${TMPDIR:-/tmp}"
echo '=== Container-only Python (180 second limit, including image verification) ==='
timeout --kill-after=10s 180s bash scripts/container_exec.sh python3 -u -c \
  'import sys; print("Container Python running:",sys.executable,flush=True)'
echo '=== Encoder Trainer import (180 second limit) ==='
timeout --kill-after=10s 180s bash scripts/container_exec.sh "$PWD/.venv-encoder/bin/python" \
  -u tools/encoder/official_train.py --check --traceback-after 30
echo 'Startup diagnostics passed; no training was run.'
