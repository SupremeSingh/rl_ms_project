#!/usr/bin/env bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
source scripts/cluster_env.sh
export MATH_RL_GPU=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
args=(--stage "${1:?}" --out "${2:?}" --phase "${3:?}" --task "${SLURM_ARRAY_TASK_ID:-0}")
if [[ "$1" == prepare ]]; then args+=(--source "${4:?}"); fi
if [[ "$1" == neural ]]; then
  export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
  srun bash scripts/container_exec.sh env -u TORCH_ALLOW_TF32_CUBLAS_OVERRIDE \
    "$PWD/.venv/bin/python" -m torch.distributed.run --standalone --nproc_per_node=2 \
    -m math_rl.fixed_policy "${args[@]}"
else
  srun bash scripts/container_exec.sh env -u TORCH_ALLOW_TF32_CUBLAS_OVERRIDE \
    "$PWD/.venv/bin/python" -m math_rl.fixed_policy "${args[@]}"
fi
