#!/usr/bin/env bash
#SBATCH --job-name=math-rl-stage1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project directory}"
export MATH_RL_GPU=1
[[ -x .venv-verifier/bin/python ]] || { echo 'Create .venv-verifier first (README).' >&2; exit 1; }
PYTHONPATH="$PWD/src" .venv-verifier/bin/python -c 'from math_rl.math_verify_reward import provenance; provenance()'
run="outputs/stage1-${SLURM_JOB_ID}"
srun bash scripts/container_exec.sh "$PWD/.venv/bin/python" scripts/preflight.py
srun bash scripts/container_exec.sh "$PWD/.venv/bin/python" scripts/generate_stage1.py --mode audit --out "$run"
srun .venv-verifier/bin/python scripts/rescore_stage1.py "$run"
echo "Review: python3 scripts/review_final_answers.py $run"
