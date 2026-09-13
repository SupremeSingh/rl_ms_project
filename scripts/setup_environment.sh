#!/usr/bin/env bash
# Run INSIDE the pinned container, in a Slurm compute allocation.
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Run setup in a compute allocation.' >&2; exit 1; }
python3 -m venv --system-site-packages --without-pip .venv
source .venv/bin/activate
curl -sS https://bootstrap.pypa.io/get-pip.py | python
python -m pip install --no-deps 'verl @ git+https://github.com/verl-project/verl.git@8d9e350ea58c7ad4b50dd14d9dcb50577242c55f'
python -m pip install -e '.[dev]'
if ! python -m pip check > configs/pip-check.txt 2>&1; then
  echo 'Inherited dependency conflicts recorded in configs/pip-check.txt; runtime checks follow.' >&2
fi
python -m pip freeze > configs/environment.txt
python -m pytest -q
python scripts/prepare_data.py
python -m math_rl.train --cfg job --resolve
