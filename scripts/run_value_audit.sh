#!/usr/bin/env bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
source scripts/cluster_env.sh
stage="${1:?}"
out="${2:?}"
phase="${3:?}"
gpus="${4:?}"
export MATH_RL_GPU=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
if [[ "$gpus" != 0 ]]; then export MATH_RL_GPU=1; fi
args=(--stage "$stage" --out "$out" --phase "$phase" --task "${SLURM_ARRAY_TASK_ID:-0}")
if [[ "$stage" == prepare ]]; then
  args+=(--source "${5:?}" --updates "${6:?}" --actor-seed "${7:?}")
fi
srun bash scripts/container_exec.sh env -u TORCH_ALLOW_TF32_CUBLAS_OVERRIDE \
  "$PWD/.venv/bin/python" -m math_rl.value_audit "${args[@]}"
