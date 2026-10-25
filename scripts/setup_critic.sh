#!/usr/bin/env bash
# Offline critic environment; run inside the pinned container.
# Keep the VERL training environment independent.
set -euo pipefail
# Debian's container Python lacks ensurepip. Create the environment without it,
# then use the existing PPO environment's pip solely as an installer targeting
# the NEW environment (the same mechanism as setup_ppo.sh).
[[ -x .venv/bin/python ]] || { echo 'Existing .venv/bin/python is required; run the project setup first.' >&2; exit 1; }
python3 -m venv --system-site-packages --without-pip .venv-critic
critic_pip=(.venv/bin/python -m pip --python "$PWD/.venv-critic/bin/python")
# Use public PyPI only; the container's optional NVIDIA index may not resolve.
export PIP_CONFIG_FILE=/dev/null
export PIP_INDEX_URL=https://pypi.org/simple
export PIP_EXTRA_INDEX_URL=''
# Record the actual inherited module, not a downloadable CUDA wheel constraint.
python3 -c 'import json, torch; from pathlib import Path; print(json.dumps(dict(version=torch.__version__, path=str(Path(torch.__file__).resolve()))))' > .venv-critic/container-torch.json
"${critic_pip[@]}" install -r requirements-critic.txt
"${critic_pip[@]}" install --no-deps -e .
.venv-critic/bin/python - <<'PY_CHECK'
import json
from pathlib import Path
import torch
import transformers
expected = json.loads(Path('.venv-critic/container-torch.json').read_text())
actual = dict(version=torch.__version__, path=str(Path(torch.__file__).resolve()))
if actual != expected:
    raise RuntimeError(f'Critic environment must reuse container PyTorch: {actual} != {expected}')
print(torch.__version__, transformers.__version__)
print('Offline critic setup complete; container PyTorch preserved.')
PY_CHECK
"${critic_pip[@]}" freeze > .venv-critic/environment.txt
