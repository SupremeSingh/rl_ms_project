#!/usr/bin/env bash
#SBATCH --job-name=math-rl-jspace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-jspace-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
source scripts/cluster_env.sh
export MATH_RL_GPU=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
[[ -x .venv-jlens/bin/python ]] || { echo 'Run setup_jlens.sh inside container first.' >&2; exit 1; }
bash scripts/container_exec.sh "$PWD/.venv-jlens/bin/python" -m pytest -q tests/test_jspace.py
srun bash scripts/container_exec.sh "$PWD/.venv-jlens/bin/python" scripts/compare_jspace_probes.py \
  --out "outputs/jspace-${SLURM_JOB_ID:?}" "$@"
