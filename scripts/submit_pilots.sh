#!/usr/bin/env bash
#SBATCH --job-name=math-rl-pilots
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=03:00:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=slurm-pilots-%j.out

# One allocation, two concurrent and resource-exclusive Slurm steps.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project directory}"
source scripts/cluster_env.sh
run="outputs/pilots-${SLURM_JOB_ID:?Run this script with sbatch}"
mkdir -p "$run"
setup_exit=not_started
ppo_exit=not_started
grpo_exit=not_started
ppo_report=not_started
grpo_report=not_started
pids=()

save_status() {
  printf 'setup_exit=%s\nppo_exit=%s\ngrpo_exit=%s\nppo_report_exit=%s\ngrpo_report_exit=%s\n' \
    "$setup_exit" "$ppo_exit" "$grpo_exit" "$ppo_report" "$grpo_report" \
    > "$run/status.txt.tmp"
  mv "$run/status.txt.tmp" "$run/status.txt"
}

stop_steps() {
  if ((${#pids[@]})); then kill "${pids[@]}" 2>/dev/null || true; fi
  exit 143
}
trap save_status EXIT
trap stop_steps TERM INT
save_status

# Finish all shared environment writes before starting either trainer.
setup_exit=running
save_status
if MATH_RL_GPU=0 bash scripts/container_exec.sh bash scripts/setup_ppo.sh > "$run/setup.log" 2>&1; then
  setup_exit=0
else
  setup_exit=$?
  echo "Setup failed; see $run/setup.log"
  exit 1
fi

launch() {
  local method=$1
  # Short, distinct paths keep Ray's Unix socket names below the OS limit.
  local ray_tmp="/tmp/mrl-${SLURM_JOB_ID}-${method}"
  mkdir -p "$ray_tmp"
  exec srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=8 \
    --gres=gpu:a5000:2 --mem=128G \
    env MATH_RL_GPU=1 RAY_ADDRESS=local RAY_TMPDIR="$ray_tmp" \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    bash scripts/container_exec.sh "$PWD/.venv/bin/python" -m math_rl.train \
    --config-name "$method" "trainer.default_local_dir=checkpoints/${method}-${SLURM_JOB_ID}" \
    "hydra.run.dir=$run/${method}-hydra" \
    > "$run/${method}.log" 2>&1
}

ppo_exit=running
grpo_exit=running
save_status
launch ppo &
pids+=("$!")
launch grpo &
pids+=("$!")
echo "PPO and GRPO launched. Logs and final status: $run"

# A failed pilot must not cancel its independent companion.
if wait "${pids[0]}"; then ppo_exit=0; else ppo_exit=$?; fi
save_status
if wait "${pids[1]}"; then grpo_exit=0; else grpo_exit=$?; fi
pids=()
save_status

ppo_report=0
python3 scripts/report_ppo.py "checkpoints/ppo-${SLURM_JOB_ID}" \
  > "$run/ppo-report.log" 2>&1 || ppo_report=$?
grpo_report=0
python3 scripts/report_ppo.py "checkpoints/grpo-${SLURM_JOB_ID}" \
  > "$run/grpo-report.log" 2>&1 || grpo_report=$?
save_status
cat "$run/status.txt"
echo 'All exit codes must be 0. Reports check execution, not improved accuracy.'
[[ "$ppo_exit" == 0 && "$grpo_exit" == 0 && "$ppo_report" == 0 && "$grpo_report" == 0 ]]
