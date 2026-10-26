#!/usr/bin/env bash
# Submit the prepared final study: smoke, or --full comparison and report.
set -euo pipefail
out='' full=0 concurrency=6 smoke_first=0 exclude_nodes='compsci-cluster-fitz-15'
while (($#)); do
  case "$1" in
    --out) out="${2:?}"; shift 2;;
    --full) full=1; shift;;
    --smoke-first) smoke_first=1; shift;;
    --concurrency) concurrency="${2:?}"; shift 2;;
    --exclude) exclude_nodes="${2:?}"; shift 2;;
    *) echo "Unknown option: $1" >&2; exit 2;;
  esac
done
[[ -n "$out" && -f scripts/numina_online.py ]] || { echo 'Run from repository root with --out.' >&2; exit 2; }
[[ "$concurrency" =~ ^[1-9][0-9]*$ ]] || { echo "Concurrency must be positive" >&2; exit 2; }
mkdir -p "$out/logs"
out="$(cd "$out" && pwd)"
[[ -f "$out/manifest.json" ]] || { echo 'Prepare or import a study first.' >&2; exit 2; }
wall=12:00:00
submit() {
  local stage="$1" dep="$2" array="$3" gpu="$4" wall="$5"
  local opts=(--parsable --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=128G
    --time="$wall" --job-name="numina-online-$stage" --output="$out/logs/$stage-%A_%a.out")
  if [[ "$gpu" == 1 ]]; then opts+=(--partition=compsci-gpu --gres=gpu:a5000:2)
  else opts+=(--partition=compsci); fi
  [[ -z "$dep" ]] || opts+=(--dependency="afterok:$dep" --kill-on-invalid-dep=yes)
  [[ -z "$array" ]] || opts+=(--array="$array")
  [[ -z "$exclude_nodes" ]] || opts+=(--exclude="$exclude_nodes")
  local job
  # Bash command substitutions may disable errexit inside this function.
  # Explicitly propagate failure; never turn a missing ID into no dependency.
  if ! job="$(sbatch "${opts[@]}" scripts/run_numina_online.sh "$stage" "$out" "$gpu")"; then
    echo "Submission failed for $stage; stopping the chain. Check squeue/sacct before retrying: a timeout may still have created a job." >&2
    return 1
  fi
  if [[ ! "$job" =~ ^[0-9]+(\;[a-zA-Z0-9_.-]+)?$ ]]; then
    echo "Invalid sbatch response for $stage: '$job'. Stopping; check Slurm before retrying." >&2
    return 1
  fi
  job="${job%%;*}"
  printf '%s\t%s\n' "$stage" "$job" >> "$out/jobs.tsv" || return 1
  printf 'Submitted %s: %s\n' "$stage" "$job" >&2
  printf '%s' "$job"
}
# Persistent marker: an ambiguous timeout must never cause duplicate submissions.
# Separate prepare/smoke and full chains each get their own marker.
marker="$out/.submission-$full"
if ! mkdir "$marker"; then
  echo "This submission chain was already attempted. Check jobs.tsv and Slurm before recovery; do not blindly resubmit." >&2
  exit 1
fi
if [[ "$full" == 0 ]]; then
  job="$(submit smoke '' '' 1 12:00:00)"
  echo "Inspect $out/smoke.json after job $job completes."
else
  dep=''
  if [[ "$smoke_first" == 1 ]]; then
    dep="$(submit smoke '' '' 1 12:00:00)"
  else
    python3 -c 'import json,sys; assert json.load(open(sys.argv[1]))["passed"]' "$out/smoke.json"
  fi
  if [[ ! -f "$out/selection.json" ]]; then dep="$(submit calibrate "$dep" '' 1 "$wall")"; fi
  last="$(python3 -c 'import json,sys; m=json.load(open(sys.argv[1])); print(len(m["seeds"])*len(m["conditions"])-1)' "$out/manifest.json")"
  train="$(submit run "$dep" "0-$last%$concurrency" 1 "$wall")"
  job="$(submit report "$train" '' 0 01:00:00)"
  echo "Final report: $out/report.txt (job $job)"
fi
