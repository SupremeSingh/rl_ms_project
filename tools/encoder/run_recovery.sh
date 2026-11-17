#!/usr/bin/env bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
source scripts/cluster_env.sh
stage=$1
out=$2
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
python="$PWD/.venv-encoder/bin/python"
if [[ "$stage" == fit || "$stage" == mc ]]; then export MATH_RL_GPU=1; fi
if [[ "$stage" == mc ]]; then python="$PWD/.venv/bin/python"; fi
if [[ -n "${SLURM_TMPDIR:-}" && -d "$SLURM_TMPDIR" ]]; then export TMPDIR="$SLURM_TMPDIR"; fi
exec bash scripts/container_exec.sh "$python" -u tools/encoder/recover_compare.py \
    --stage "$stage" --out "$out" --task "${SLURM_ARRAY_TASK_ID:-0}"
