#!/usr/bin/env bash
# CPU-only readout of completed offline predictions. No generation or training.
set -euo pipefail
source_study="${1:?Usage: bash scripts/submit_probe_poster.sh SOURCE SPARSE --out DIR}"
sparse="${2:?Supply completed sparse study}"
shift 2
out=''
while (($#)); do
  case "$1" in
    --out) out="${2:?}"; shift 2;;
    *) echo "Unknown option: $1" >&2; exit 2;;
  esac
done
[[ -n "$out" && -f scripts/export_probe_poster.py && -f "$source_study/summary.json" && -f "$sparse/predictions.npz" ]] || { echo 'Run from repository root with completed studies and --out.' >&2; exit 2; }
source_study="$(cd "$source_study" && pwd)"
sparse="$(cd "$sparse" && pwd)"
out="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$out")"
[[ "$out" != "$source_study" && "$out" != "$source_study/"* && "$out" != "$sparse" && "$out" != "$sparse/"* ]] || { echo 'Output must be separate from source studies.' >&2; exit 2; }
mkdir -p "$out"
[[ -z "$(ls -A "$out")" ]] || { echo 'Output must be empty; use a new directory.' >&2; exit 2; }
job="$(sbatch --parsable --partition=compsci --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G \
  --time=02:00:00 --job-name=probe-poster --output="$out.slurm-%j.out" \
  scripts/run_probe_poster.sh "$source_study" "$sparse" "$out")"
echo "Submitted CPU evidence export: ${job%%;*}"
echo "After completion: cat $out/report.md"
