#!/usr/bin/env bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
source scripts/cluster_env.sh
export MATH_RL_GPU="${5:?}" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
# Setup is deliberately separate: concurrent jobs must not mutate the shared venv.
srun bash scripts/container_exec.sh env -u TORCH_ALLOW_TF32_CUBLAS_OVERRIDE \
  "$PWD/.venv/bin/python" scripts/numina_online.py --stage "$1" --out "$2" \
  --source "$3" --sparse "$4" --preset "${6:-standard}" --task "${SLURM_ARRAY_TASK_ID:-0}"
