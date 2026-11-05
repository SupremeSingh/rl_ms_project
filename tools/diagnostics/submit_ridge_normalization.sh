#!/usr/bin/env bash
# Submit from the repository root; saved feature arrays only, no GPU required.
#SBATCH --job-name=ridge-normalization
#SBATCH --partition=compsci
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ridge-normalization-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
source scripts/cluster_env.sh
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONUNBUFFERED=1
features="${1:?Usage: sbatch tools/diagnostics/submit_ridge_normalization.sh FEATURE_SOURCE FROZEN_FILTER_STUDY NEW_OUTPUT}"
study="${2:?Missing frozen-filter study directory}"
out="${3:?Missing new output directory}"
shift 3
srun bash scripts/container_exec.sh "$PWD/.venv-critic/bin/python" \
  tools/diagnostics/ridge_normalization.py \
  --features "$features" --study "$study" --out "$out" --strength 0.0001 "$@"
