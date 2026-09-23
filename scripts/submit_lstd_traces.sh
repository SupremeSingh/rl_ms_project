#!/usr/bin/env bash
#SBATCH --job-name=math-rl-lstd-traces
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-lstd-traces-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository}"
source scripts/cluster_env.sh
export MATH_RL_GPU=0 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" PYTHONUNBUFFERED=1
srun bash scripts/container_exec.sh "$PWD/.venv/bin/python" scripts/compare_lstd_traces.py \
  --out "outputs/lstd-traces-${SLURM_JOB_ID:?Run with sbatch}" "$@"
