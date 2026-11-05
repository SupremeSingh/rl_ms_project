#!/usr/bin/env bash
# Explicit stages: mandatory smoke first, then a 21-task matched array.
#SBATCH --job-name=slice-online-critics
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --exclude=compsci-cluster-fitz-15
#SBATCH --output=logs/slice-online-critics-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
source scripts/cluster_env.sh
stage="${1:?Usage: sbatch ... smoke|run|report STUDY_DIRECTORY}"
out="${2:?Missing study directory}"
export MATH_RL_GPU=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONUNBUFFERED=1
if [[ "$stage" == report || "$stage" == smoke-gate ]]; then export MATH_RL_GPU=0; fi
srun bash scripts/container_exec.sh env -u TORCH_ALLOW_TF32_CUBLAS_OVERRIDE \
 "$PWD/.venv/bin/python" tools/diagnostics/online_critics.py \
 --stage "$stage" --out "$out" --task "${SLURM_ARRAY_TASK_ID:-0}"
