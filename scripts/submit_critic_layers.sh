#!/usr/bin/env bash
#SBATCH --job-name=math-rl-layers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-critic-layers-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project directory}"
source scripts/cluster_env.sh
export MATH_RL_GPU=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
bash scripts/container_exec.sh "$PWD/.venv/bin/python" -m pytest -q tests/test_critic_layers.py
srun bash scripts/container_exec.sh "$PWD/.venv/bin/python" scripts/compare_critic_layers.py \
  --out "outputs/critic-layers-${SLURM_JOB_ID:?}" "$@"
