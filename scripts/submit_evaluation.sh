#!/usr/bin/env bash
#SBATCH --job-name=math-rl-evaluation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-evaluation-%j.out

# Usage: sbatch scripts/submit_evaluation.sh PILOT_JOB_ID
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project directory}"
[[ $# == 1 && "$1" =~ ^[0-9]+$ ]] || { echo 'Usage: sbatch scripts/submit_evaluation.sh PILOT_JOB_ID' >&2; exit 1; }
source scripts/cluster_env.sh
export MATH_RL_GPU=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
run="outputs/evaluation-${SLURM_JOB_ID:?Run with sbatch}"
srun bash scripts/container_exec.sh "$PWD/.venv/bin/python" scripts/evaluate_pilots.py \
  run --pilot-job "$1" --out "$run"
echo "Finished: $run/summary.json (accuracy), $run/changes.jsonl (answer changes), $run/review.txt (sample)"
