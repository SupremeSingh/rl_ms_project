#!/usr/bin/env bash
#SBATCH --job-name=slice-ppo-smoke
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --exclude=compsci-cluster-fitz-15
#SBATCH --output=logs/slice-ppo-smoke-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
source scripts/cluster_env.sh
export MATH_RL_GPU=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
artifact="${1:?Usage: sbatch scripts/submit_slice_ppo_smoke.sh ARTIFACT NEW_OUTPUT_DIRECTORY}"
out="${2:?Pass a new output directory}"
srun bash scripts/container_exec.sh env -u TORCH_ALLOW_TF32_CUBLAS_OVERRIDE \
  "$PWD/.venv/bin/python" scripts/smoke_slice_ppo.py "$artifact" --out "$out"
