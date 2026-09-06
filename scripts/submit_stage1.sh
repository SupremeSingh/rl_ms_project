#!/usr/bin/env bash
#SBATCH --job-name=math-rl-stage1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from math-rl}"
: "${MATH_RL_IMAGE:?Set the absolute SIF image path}"
srun apptainer exec --nv --bind "$PWD:$PWD" "$MATH_RL_IMAGE" \
  "$PWD/.venv/bin/python" scripts/generate_stage1.py "$@"
