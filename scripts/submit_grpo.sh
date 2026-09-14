#!/usr/bin/env bash
#SBATCH --job-name=math-rl-grpo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project directory}"
export MATH_RL_GPU=1
srun bash scripts/container_exec.sh "$PWD/.venv/bin/python" -m math_rl.train \
  --config-name grpo "trainer.default_local_dir=checkpoints/grpo-${SLURM_JOB_ID}" "$@"
