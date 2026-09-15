#!/usr/bin/env bash
#SBATCH --job-name=math-rl-critics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-critics-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository}"
source scripts/cluster_env.sh
export MATH_RL_GPU=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
srun bash scripts/container_exec.sh "$PWD/.venv/bin/python" scripts/frozen_critics.py \
  --out "outputs/critics-${SLURM_JOB_ID:?Run with sbatch}" "$@"
