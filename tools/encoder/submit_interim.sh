#!/usr/bin/env bash
# Submit from the repository root. Optional third argument: extraction array ID.
set -euo pipefail
study=${1:-outputs/encoder-vine-unpadded-numina-v1}
out=${2:-outputs/encoder-vine-unpadded-interim-42-v1}
extract_job=${3:-13771037}
if [[ -e "$out" ]]; then
  echo "Use a fresh interim output directory: $out" >&2
  exit 1
fi
mkdir -p "$out/logs"
dependency=afterok
for task in {8..15}; do dependency+=":${extract_job}_${task}"; done
fit_job=$(sbatch --parsable --partition=compsci-gpu --gres=gpu:a5000:1 \
  --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=64G --time=24:00:00 \
  --array=0-23%2 --job-name=encoder-interim-fit \
  --dependency="$dependency" --kill-on-invalid-dep=yes \
  --output="$out/logs/fit-%A_%a.out" \
  tools/encoder/run_interim.sh fit "$study" "$out")
fit_job=${fit_job%%;*}
echo "Submitted interim fitting: $fit_job"
report_job=$(sbatch --parsable --partition=compsci-gpu --gres=gpu:a5000:1 \
  --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=00:30:00 \
  --job-name=encoder-interim-report --dependency="afterok:$fit_job" \
  --kill-on-invalid-dep=yes --output="$out/logs/report-%j.out" \
  tools/encoder/run_interim.sh report "$study" "$out")
printf 'fit\t%s\nreport\t%s\n' "$fit_job" "$report_job" > "$out/jobs.tsv"
echo "Submitted report: $report_job"
echo "Report: $out/report.txt"
