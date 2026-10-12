#!/usr/bin/env bash
# Submit a dependency graph, not a monolithic allocation. Run with bash, not sbatch.
set -euo pipefail
out='' mode='full' resume='prepare'
while (($#)); do
  case "$1" in
    --out) out="$2"; shift 2 ;;
    --smoke) mode='smoke'; shift ;;
    --full) mode='full'; shift ;;
    --resume-from) resume="$2"; shift 2 ;;
    *) echo "Usage: bash scripts/submit_large_jlens.sh --full|--smoke --out DIR [--resume-from prepare|screen|lens|generate|extract|pack|fit|linear-stats|linear-fit|report]" >&2; exit 2 ;;
  esac
done
[[ -n "$out" ]] || { echo 'Give a stable --out directory for resumable artifacts.' >&2; exit 2; }
[[ -f scripts/large_jlens_study.py ]] || { echo 'Run from the project root.' >&2; exit 2; }
for python in .venv/bin/python .venv-jlens/bin/python .venv-ppo-verifier/bin/python; do
  [[ -x "$python" ]] || { echo "Missing $python; follow SETUP.MD." >&2; exit 2; }
done
stages=(prepare screen lens generate extract pack fit linear-stats linear-fit report)
start=-1
for i in "${!stages[@]}"; do [[ "${stages[$i]}" != "$resume" ]] || start="$i"; done
((start >= 0)) || { echo 'Invalid resume stage' >&2; exit 2; }
mkdir -p "$out/logs"
out="$(cd "$out" && pwd)"
if [[ -f "$out/submission-mode.txt" ]]; then
  [[ "$(cat "$out/submission-mode.txt")" == "$mode" ]] || { echo 'Smoke/full mode differs from existing study.' >&2; exit 2; }
else
  printf '%s\n' "$mode" > "$out/submission-mode.txt"
fi
if [[ "$resume" == prepare && -f "$out/jobs.tsv" ]]; then
  echo 'Existing submissions: inspect jobs.tsv; use --resume-from after stopping old dependent jobs.' >&2
  exit 2
fi
submit() {
  local stage="$1" dependency="$2" array="$3" gpu="$4" duration="$5"
  local opts=(--parsable --job-name="jlens-large-$stage" --nodes=1 --ntasks=1
    --cpus-per-task=8 --mem=96G --time="$duration" --output="$out/logs/$stage-%A_%a.out")
  if [[ "$gpu" == 1 ]]; then
    opts+=(--partition=compsci-gpu --gres=gpu:a5000:1)
  else
    opts+=(--partition=compsci)
  fi
  [[ -z "$dependency" ]] || opts+=(--dependency="afterok:$dependency" --kill-on-invalid-dep=yes)
  [[ -z "$array" ]] || opts+=(--array="$array")
  local job
  job="$(sbatch "${opts[@]}" scripts/run_large_jlens_stage.sh "$out" "$stage" "$mode" "$gpu")"
  job="${job%%;*}"
  printf '%s\t%s\n' "$stage" "$job" >> "$out/jobs.tsv"
  printf 'Submitted %s: %s\n' "$stage" "$job" >&2
  printf '%s' "$job"
}
dependency=''; lens_job=''; generation_job=''
((start > 0)) || dependency="$(submit prepare '' '' 0 06:00:00)"
((start > 1)) || dependency="$(submit screen "$dependency" '' 1 12:00:00)"
((start > 2)) || lens_job="$(submit lens "$dependency" '' 1 2-00:00:00)"
((start > 3)) || generation_job="$(submit generate "$dependency" '0-3%4' 1 2-00:00:00)"
if ((start <= 4)); then
  dependencies="${generation_job}${generation_job:+${lens_job:+:}}${lens_job}"
  dependency="$(submit extract "$dependencies" '0-3%4' 1 2-00:00:00)"
fi
((start > 5)) || dependency="$(submit pack "$dependency" '0-4%2' 0 1-00:00:00)"
fit_job=''; linear_job=''
((start > 6)) || fit_job="$(submit fit "$dependency" '0-14%3' 1 2-00:00:00)"
((start > 7)) || linear_job="$(submit linear-stats "$dependency" '0-3%4' 1 2-00:00:00)"
((start > 8)) || linear_job="$(submit linear-fit "$linear_job" '0-4%2' 0 1-00:00:00)"
dependencies="${fit_job}${fit_job:+${linear_job:+:}}${linear_job}"
dependency="$(submit report "$dependencies" '' 1 06:00:00)"
printf '\nStudy: %s\nJob list: %s/jobs.tsv\nFinal report job: %s\n' "$out" "$out" "$dependency"
