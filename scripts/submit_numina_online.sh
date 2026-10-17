#!/usr/bin/env bash
# Submit preparation + smoke; --full submits calibration + 45 runs + report.
set -euo pipefail
source_study="${1:?Usage: bash scripts/submit_numina_online.sh SOURCE SPARSE --out DIR [--large] [--full]}"
sparse="${2:?Supply the completed 768-dimensional sparse study}"
shift 2
out='' full=0 preset='' concurrency=2
while (($#)); do
  case "$1" in
    --out) out="${2:?}"; shift 2;;
    --full) full=1; shift;;
    --concurrency) concurrency="${2:?}"; shift 2;;
    --large) preset=large; shift;;
    *) echo "Unknown option: $1" >&2; exit 2;;
  esac
done
[[ -n "$out" && -f scripts/numina_online.py ]] || { echo 'Run from repository root with --out.' >&2; exit 2; }
[[ "$concurrency" =~ ^[1-9][0-9]*$ ]] || { echo "Concurrency must be positive" >&2; exit 2; }
source_study="$(cd "$source_study" && pwd)"
sparse="$(cd "$sparse" && pwd)"
mkdir -p "$out"
out="$(cd "$out" && pwd)"
[[ "$out" != "$source_study" && "$out" != "$source_study/"* && "$out" != "$sparse" && "$out" != "$sparse/"* ]] || { echo 'Separate output required.' >&2; exit 2; }
mkdir -p "$out/logs"
if [[ -f "$out/manifest.json" ]]; then
  saved_preset="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("preset","standard"))' "$out/manifest.json")"
  [[ -z "$preset" || "$preset" == "$saved_preset" ]] || { echo 'Budget changed: use a new output directory.' >&2; exit 2; }
  preset="$saved_preset"
fi
preset="${preset:-standard}"
wall=12:00:00
[[ "$preset" != large ]] || wall=1-00:00:00
submit() {
  local stage="$1" dep="$2" array="$3" gpu="$4" wall="$5"
  local opts=(--parsable --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=128G
    --time="$wall" --job-name="numina-online-$stage" --output="$out/logs/$stage-%A_%a.out")
  if [[ "$gpu" == 1 ]]; then opts+=(--partition=compsci-gpu --gres=gpu:a5000:2)
  else opts+=(--partition=compsci); fi
  [[ -z "$dep" ]] || opts+=(--dependency="afterok:$dep" --kill-on-invalid-dep=yes)
  [[ -z "$array" ]] || opts+=(--array="$array")
  local job
  job="$(sbatch "${opts[@]}" scripts/run_numina_online.sh "$stage" "$out" "$source_study" "$sparse" "$gpu" "$preset")"
  job="${job%%;*}"
  printf '%s\t%s\n' "$stage" "$job" >> "$out/jobs.tsv"
  printf 'Submitted %s: %s\n' "$stage" "$job" >&2
  printf '%s' "$job"
}
if [[ "$full" == 0 ]]; then
  dep=''
  if [[ ! -f "$out/manifest.json" ]]; then dep="$(submit prepare '' '' 0 03:00:00)"; fi
  job="$(submit smoke "$dep" '' 1 12:00:00)"
  echo "Inspect $out/smoke.json after job $job completes."
else
  python3 -c 'import json,sys; assert json.load(open(sys.argv[1]))["passed"]' "$out/smoke.json"
  dep=''
  if [[ ! -f "$out/selection.json" ]]; then dep="$(submit calibrate '' '' 1 "$wall")"; fi
  last="$(python3 -c 'import json,sys; m=json.load(open(sys.argv[1])); print(len(m["seeds"])*len(m["conditions"])-1)' "$out/manifest.json")"
  train="$(submit run "$dep" "0-$last%$concurrency" 1 "$wall")"
  job="$(submit report "$train" '' 0 03:00:00)"
  echo "Final report: $out/report.txt (job $job)"
fi
