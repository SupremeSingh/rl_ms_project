#!/usr/bin/env bash
#SBATCH --job-name=slice-cost-audit
#SBATCH --partition=compsci
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slice-cost-audit-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
source scripts/cluster_env.sh
export MATH_RL_GPU=0
study="${1:?Usage: sbatch tools/transfer/submit_cost_audit.sh STUDY NEW_AUDIT_OUTPUT [--extra-study PATH ...]}"
out="${2:?Missing new audit output directory}"
shift 2
# Host Python retains access to sacct; this audit needs only the standard library.
python3 tools/transfer/cost_audit.py --study "$study" --out "$out" "$@"
# Only plotting uses the container's scientific Python environment. No GPU requested.
bash scripts/container_exec.sh "$PWD/.venv-critic/bin/python" \
  tools/transfer/cost_audit.py --plot-report "$out/summary.json" --out "$out/figures"
