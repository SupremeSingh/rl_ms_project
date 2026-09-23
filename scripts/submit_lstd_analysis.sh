#!/usr/bin/env bash
#SBATCH --job-name=math-rl-lstd-analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-lstd-analysis-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository}"
source scripts/cluster_env.sh
export MATH_RL_GPU=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
srun bash scripts/container_exec.sh "$PWD/.venv/bin/python" scripts/analyze_lstd.py \
  --out "outputs/lstd-analysis-${SLURM_JOB_ID:?Run with sbatch}" "$@"
