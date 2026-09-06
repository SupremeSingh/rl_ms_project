#!/usr/bin/env bash
# Submit all three jobs; Slurm enforces the split-run dependency.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs
control=checkpoints/resume-control
split=checkpoints/resume-split
[[ ! -e "$control" && ! -e "$split" ]] || { echo 'Preserve existing comparison directories before starting a new comparison.' >&2; exit 1; }
sbatch scripts/submit_smoke.sh trainer.default_local_dir="$control" trainer.total_training_steps=4
job=$(sbatch --parsable scripts/submit_smoke.sh trainer.default_local_dir="$split" trainer.total_training_steps=2)
sbatch --dependency="afterok:${job%%;*}" scripts/submit_smoke.sh \
  trainer.default_local_dir="$split" trainer.resume_mode=resume_path \
  trainer.resume_from_path="$split/global_step_2" trainer.total_training_steps=4
