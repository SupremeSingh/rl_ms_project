#!/usr/bin/env bash
# Validation-only by default. Explicit --stage evaluate is a separate submission.
#SBATCH --job-name=slice-mlp-tune
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=compsci-cluster-fitz-15
#SBATCH --output=logs/slice-mlp-tune-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
source scripts/cluster_env.sh
export MATH_RL_GPU=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONUNBUFFERED=1
features="${1:?Usage: sbatch tools/diagnostics/submit_tune_slice_mlp.sh FEATURE_SOURCE FILTER_STUDY NEW_OUTPUT [--stage tune|evaluate]}"
study="${2:?Missing filter study}"
out="${3:?Missing output directory}"
shift 3
srun bash scripts/container_exec.sh env -u TORCH_ALLOW_TF32_CUBLAS_OVERRIDE \
  "$PWD/.venv-critic/bin/python" tools/diagnostics/tune_slice_mlp.py \
  --features "$features" --study "$study" --out "$out" "$@"
