#!/usr/bin/env bash
#SBATCH --job-name=math-rl-lstd-math
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-lstd-math-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository}"
source scripts/cluster_env.sh
export MATH_RL_GPU=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
srun bash scripts/container_exec.sh "$PWD/.venv/bin/python" scripts/validate_lstd_seed.py \
  --out "outputs/lstd-math-${SLURM_JOB_ID:?Run with sbatch}" --math-hard "$@"
