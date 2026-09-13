#!/usr/bin/env bash
#SBATCH --job-name=math-rl-diagnostic
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
srun bash scripts/container_exec.sh "$PWD/.venv/bin/python" scripts/preflight.py
for contract in answer boxed; do
    for tokens in 512 2048; do
        srun bash scripts/container_exec.sh "$PWD/.venv/bin/python" scripts/generate_stage1.py \
            --mode diagnostic --contract "$contract" --max-tokens "$tokens" \
            --out "outputs/diagnostic-${SLURM_JOB_ID}/${contract}-${tokens}"
    done
done
