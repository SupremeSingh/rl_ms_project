#!/usr/bin/env bash
#SBATCH --job-name=math-rl-followups
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-critic-followups-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project directory}"
source scripts/cluster_env.sh
export MATH_RL_GPU=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
bash scripts/container_exec.sh bash scripts/setup_ppo.sh
bash scripts/container_exec.sh "$PWD/.venv/bin/python" -m pytest -q tests/test_cumulative_lstd.py tests/test_online_critic.py tests/test_critic_layers.py tests/test_critic_followups.py
srun bash scripts/container_exec.sh "$PWD/.venv/bin/python" scripts/critic_followups.py \
  --out "outputs/critic-followups-${SLURM_JOB_ID:?}" "$@"
