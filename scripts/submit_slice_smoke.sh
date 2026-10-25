#!/usr/bin/env bash
# Submit with sbatch, from the repository root. Environments must be installed first.
#SBATCH --job-name=slice-package-smoke
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --exclude=compsci-cluster-fitz-15
#SBATCH --output=logs/slice-package-smoke-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
source scripts/cluster_env.sh
export MATH_RL_GPU=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
out="${1:?Usage: sbatch scripts/submit_slice_smoke.sh NEW_OUTPUT_DIRECTORY}"
srun bash scripts/container_exec.sh env -u TORCH_ALLOW_TF32_CUBLAS_OVERRIDE \
  "$PWD/.venv-critic/bin/python" scripts/smoke_slice_package.py \
  --source outputs/numina-online-768-large-v1/data/questions.json --out "$out"
