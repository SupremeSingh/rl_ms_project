#!/usr/bin/env bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/cluster_env.sh
export MATH_RL_GPU=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
args=(--stage "${1:?}" --out "${2:?}" --task "${SLURM_ARRAY_TASK_ID:-0}")
if [[ "$1" == prepare ]]; then args+=(--source "${3:?}"); fi
srun bash scripts/container_exec.sh env -u TORCH_ALLOW_TF32_CUBLAS_OVERRIDE \
  "$PWD/.venv/bin/python" -m math_rl.learning_study "${args[@]}"
