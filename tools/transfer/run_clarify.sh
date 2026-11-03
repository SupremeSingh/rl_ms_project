#!/usr/bin/env bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
source scripts/cluster_env.sh
stage="${1:?stage}"; out="${2:?output}"; shift 2
export MATH_RL_GPU=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONUNBUFFERED=1
if [[ "$stage" == generate || "$stage" == fit ]]; then export MATH_RL_GPU=1; fi
srun bash scripts/container_exec.sh env -u TORCH_ALLOW_TF32_CUBLAS_OVERRIDE \
 "$PWD/.venv-critic/bin/python" tools/transfer/clarify.py --stage "$stage" --out "$out" \
 --task "${SLURM_ARRAY_TASK_ID:-0}" "$@"
