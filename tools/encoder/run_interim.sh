#!/usr/bin/env bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
source scripts/cluster_env.sh
export MATH_RL_GPU=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
exec bash scripts/container_exec.sh "$PWD/.venv-encoder/bin/python" \
  tools/encoder/interim_compare.py --stage "$1" --study "$2" --out "$3" \
  --task "${SLURM_ARRAY_TASK_ID:-0}"
