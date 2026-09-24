#!/usr/bin/env bash
#SBATCH --job-name=math-rl-online
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-math-online-%j.out

# One allocation; sequential methods avoid GPU sharing and make costs comparable.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project directory}"
source scripts/cluster_env.sh
export MATH_RL_GPU=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
[[ $# -ge 1 ]] || { echo 'Usage: sbatch scripts/submit_math_comparison.sh outputs/math-fit-... [--steps 30] [--seeds 42 43 44]' >&2; exit 1; }
bash scripts/container_exec.sh bash scripts/setup_ppo.sh
bash scripts/container_exec.sh "$PWD/.venv/bin/python" -m pytest -q \
  tests/test_online_critic.py tests/test_math_comparison.py
srun bash scripts/container_exec.sh "$PWD/.venv/bin/python" scripts/math_comparison.py \
  --out "outputs/math-online-${SLURM_JOB_ID:?}" "$@"
