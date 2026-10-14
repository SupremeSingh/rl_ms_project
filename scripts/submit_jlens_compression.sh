#!/usr/bin/env bash
#SBATCH --job-name=jlens-compression
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-jlens-compression-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
source scripts/cluster_env.sh
export MATH_RL_GPU=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
source_study="${1:?Give the completed large J-lens study directory}"
shift
srun bash scripts/container_exec.sh "$PWD/.venv-jlens/bin/python" \
  scripts/compress_jlens_critics.py "$source_study" \
  --out "outputs/jlens-compression-${SLURM_JOB_ID:?}" "$@"
