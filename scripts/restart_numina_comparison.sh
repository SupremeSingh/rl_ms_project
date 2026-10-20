#!/usr/bin/env bash
# Fresh actor runs; reuse only verified data, filters and validation calibration.
set -euo pipefail
parent="${1:?Usage: bash scripts/restart_numina_comparison.sh PARENT --out NEW_DIR}"
shift
out=''
while (($#)); do
  case "$1" in
    --out) out="${2:?}"; shift 2;;
    *) echo "Unknown option: $1" >&2; exit 2;;
  esac
done
[[ -n "$out" && -f scripts/numina_online.py ]] || { echo 'Run from repository root with --out.' >&2; exit 2; }
[[ ! -e "$out" ]] || { echo 'Use a new output directory; existing results are preserved.' >&2; exit 2; }
source scripts/cluster_env.sh
bash scripts/container_exec.sh "$PWD/.venv/bin/python" scripts/numina_online.py \
  --stage short --comparison --source "$parent" --out "$out"
source_study="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["source"])' "$out/manifest.json")"
sparse="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["sparse_source"])' "$out/manifest.json")"
bash scripts/submit_numina_online.sh "$source_study" "$sparse" \
  --out "$out" --full --smoke-first --concurrency 6 \
  --exclude compsci-cluster-fitz-15
