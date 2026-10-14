#!/usr/bin/env bash
# bash submission helper; screen first, full only after inspecting screen-report.txt.
set -euo pipefail
source_study="${1:?Usage: bash scripts/submit_sparse_critics.sh SOURCE --out DIR [--width 768|1536|3072] [--full]}"
shift
out='' full=0 width=''
while (($#)); do
  case "$1" in
    --out) out="$2";shift 2;;
    --full) full=1;shift;;
    --width) width="${2:?Supply a bottleneck width}";shift 2;;
    *) echo "Unknown option: $1" >&2;exit 2;;
  esac
done
[[ -n "$out" && -f scripts/sparse_critic_study.py ]] || { echo 'Run from project root and supply --out.' >&2;exit 2; }
[[ -f "$source_study/summary.json" ]] || { echo 'Requires completed large Numina study.' >&2;exit 2; }
mkdir -p "$out/logs"
out="$(cd "$out" && pwd)"
source_study="$(cd "$source_study" && pwd)"
[[ "$out" != "$source_study" && "$out" != "$source_study/"* ]] || { echo 'Use separate output directory.' >&2;exit 2; }
# Reuse the saved width for later stages unless explicitly overridden.
if [[ -z "$width" ]]; then
  if [[ -f "$out/protocol.json" ]]; then
    width="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["width"])' "$out/protocol.json")"
  else width=768;fi
fi
case "$width" in 768|1536|3072) ;; *) echo 'Supported widths: 768, 1536, 3072.' >&2;exit 2;; esac
if [[ -f "$out/protocol.json" ]]; then
  python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d["width"] == int(sys.argv[2]) else "Width differs from saved protocol; use a separate output directory")' "$out/protocol.json" "$width"
fi
submit() {
  local stage="$1" dep="$2" array="$3" gpu="$4"
  local opts=(--parsable --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G
    --time=1-00:00:00 --job-name="sparse-critic-$stage" --output="$out/logs/$stage-%A_%a.out")
  if [[ "$gpu" == 1 ]];then opts+=(--partition=compsci-gpu --gres=gpu:a5000:1)
  else opts+=(--partition=compsci);fi
  [[ -z "$dep" ]] || opts+=(--dependency="afterok:$dep" --kill-on-invalid-dep=yes)
  [[ -z "$array" ]] || opts+=(--array="$array")
  local job
  job="$(sbatch "${opts[@]}" scripts/run_sparse_critics.sh "$source_study" "$out" "$stage" "$gpu" "$width")"
  job="${job%%;*}"
  printf '%s\t%s\n' "$stage" "$job" >> "$out/jobs.tsv"
  printf 'Submitted %s: %s\n' "$stage" "$job" >&2
  printf '%s' "$job"
}
if [[ "$full" == 0 ]]; then
  job="$(submit screen '' '' 1)"
  echo "Inspect $out/screen-report.txt when $job finishes."
else
  [[ -f "$out/screen.json" ]] || { echo 'Run the validation screen first.' >&2;exit 2; }
  python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d["recommended_full"] else "Screen did not pass; inspect screen-report.txt")' "$out/screen.json"
  shards="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["config"]["shards"]-1)' "$source_study/manifest.json")"
  job="$(submit stats '' "0-$shards%4" 1)"
  report="$(submit report "$job" '' 0)"
  echo "Final report job: $report; output: $out/report.txt"
fi
