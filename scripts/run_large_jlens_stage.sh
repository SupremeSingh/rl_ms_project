#!/usr/bin/env bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
out="${1:?}" stage="${2:?}" mode="${3:?}" gpu="${4:?}"
source scripts/cluster_env.sh
export MATH_RL_GPU="$gpu" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
python="$PWD/.venv-jlens/bin/python"
case "$stage" in prepare|screen|generate) python="$PWD/.venv/bin/python" ;; esac
args=(--out "$out" --stage "$stage" --task "${SLURM_ARRAY_TASK_ID:-0}")
[[ "$mode" != smoke ]] || args+=(--smoke)
srun bash scripts/container_exec.sh env -u TORCH_ALLOW_TF32_CUBLAS_OVERRIDE \
  "$python" scripts/large_jlens_study.py "${args[@]}"
