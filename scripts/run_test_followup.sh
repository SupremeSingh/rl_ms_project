#!/usr/bin/env bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/cluster_env.sh
export MATH_RL_GPU=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
module=${1:?}; stage=${2:?}; out=${3:?}; source_path=${4:-}; smoke_path=${5:-}
args=(--stage "$stage" --out "$out")
if [[ "$module" == test_followup ]]; then
  args+=(--task "${SLURM_ARRAY_TASK_ID:-0}")
fi
if [[ "$stage" == prepare ]]; then
  args+=(--source "$source_path")
  if [[ "$module" == cost_benchmark ]]; then
    if [[ -z "$smoke_path" ]]; then args+=(--quick); else args+=(--passed-smoke "$smoke_path"); fi
  fi
fi
srun bash scripts/container_exec.sh env -u TORCH_ALLOW_TF32_CUBLAS_OVERRIDE \
  "$PWD/.venv/bin/python" -m "math_rl.$module" "${args[@]}"
