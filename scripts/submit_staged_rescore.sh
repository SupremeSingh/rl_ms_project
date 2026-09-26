#!/usr/bin/env bash
#SBATCH --job-name=math-rl-staged-rescore
#SBATCH --partition=compsci
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-staged-rescore-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository}"
source scripts/cluster_env.sh
export MATH_RL_GPU=0 PYTHONUNBUFFERED=1
srun bash scripts/container_exec.sh "$PWD/.venv/bin/python" scripts/rescore_staged_answers.py \
  --out "outputs/staged-rescore-${SLURM_JOB_ID:?Run with sbatch}" "$@"
