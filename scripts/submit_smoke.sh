#!/usr/bin/env bash
#SBATCH --job-name=math-rl-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the math-rl directory}"
: "${MATH_RL_IMAGE:?Set MATH_RL_IMAGE to the absolute path of your SIF image}"

# The environment is made inside this image during setup.
srun apptainer exec --nv --bind "$PWD:$PWD" "$MATH_RL_IMAGE" \
  "$PWD/.venv/bin/python" -m math_rl.train "$@"
