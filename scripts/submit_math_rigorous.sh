#!/usr/bin/env bash
#SBATCH --job-name=math-rl-rigorous
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-math-rigorous-%j.out

# One allocation; sequential methods avoid GPU sharing and make costs comparable.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project directory}"
source scripts/cluster_env.sh
export MATH_RL_GPU=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
[[ $# -ge 1 ]] || { echo 'Usage: sbatch scripts/submit_math_rigorous.sh outputs/math-fit-... [--steps 60] [--seeds 42 43 44]' >&2; exit 1; }
bash scripts/container_exec.sh bash scripts/setup_ppo.sh
bash scripts/container_exec.sh "$PWD/.venv/bin/python" -m pytest -q \
  tests/test_online_critic.py tests/test_critic_replay.py tests/test_math_comparison.py tests/test_ppo_baseline.py
srun bash scripts/container_exec.sh "$PWD/.venv/bin/python" scripts/math_rigorous.py \
  --out "outputs/math-rigorous-${SLURM_JOB_ID:?}" "$@"
