#!/usr/bin/env bash
#SBATCH --job-name=math-rl-jlens-direct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-jlens-direct-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
source scripts/cluster_env.sh
export MATH_RL_GPU=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
bash scripts/container_exec.sh env -u TORCH_ALLOW_TF32_CUBLAS_OVERRIDE \
  "$PWD/.venv-jlens/bin/python" -m pytest -q tests/test_direct_jlens.py
srun bash scripts/container_exec.sh env -u TORCH_ALLOW_TF32_CUBLAS_OVERRIDE \
  "$PWD/.venv-jlens/bin/python" scripts/compare_direct_jlens.py \
  --out "outputs/jlens-direct-${SLURM_JOB_ID:?}" "$@"
