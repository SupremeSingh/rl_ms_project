#!/usr/bin/env bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
source scripts/cluster_env.sh
export MATH_RL_GPU=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
stage=$1
out=$2
shift 2
if [[ "$stage" == extract ]]; then
  arms=(causal bidir adapted)
  set -- "$@" --arm "${arms[${SLURM_ARRAY_TASK_ID:?}]}"
fi
exec bash scripts/container_exec.sh "$PWD/.venv-encoder/bin/python" \
  tools/encoder/study.py --stage "$stage" --out "$out" "$@"
