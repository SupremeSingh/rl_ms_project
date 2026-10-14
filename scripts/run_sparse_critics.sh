#!/usr/bin/env bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
source scripts/cluster_env.sh
export MATH_RL_GPU="${4:?}" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
srun bash scripts/container_exec.sh env -u TORCH_ALLOW_TF32_CUBLAS_OVERRIDE \
  "$PWD/.venv-jlens/bin/python" scripts/sparse_critic_study.py "$1" \
  --out "$2" --stage "$3" --width "${5:-768}" --task "${SLURM_ARRAY_TASK_ID:-0}"
