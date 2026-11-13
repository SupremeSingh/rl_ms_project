#!/usr/bin/env bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
source scripts/cluster_env.sh
export MATH_RL_GPU=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
stage=$1
out=$2
shift 2
python="$PWD/.venv-encoder/bin/python"
if [[ "$stage" == generate || "$stage" == mc ]]; then python="$PWD/.venv/bin/python"; fi
if [[ "$stage" == distributed-check ]]; then
  exec bash scripts/container_exec.sh "$python" -m torch.distributed.run --standalone --nproc_per_node=4 tools/encoder/distributed_check.py --out "$out"
fi
exec bash scripts/container_exec.sh "$python" tools/encoder/compare.py --stage "$stage" --out "$out" --task "${SLURM_ARRAY_TASK_ID:-0}" "$@"
