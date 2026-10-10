#!/usr/bin/env bash
# Run inside our pinned container, not inside the PPO virtual environment.
set -euo pipefail
# Debian's container Python lacks ensurepip. Create the environment without it,
# then use the existing PPO environment's pip solely as an installer targeting
# the NEW environment (the same mechanism as setup_ppo.sh).
[[ -x .venv/bin/python ]] || { echo 'Existing .venv/bin/python is required; run the project setup first.' >&2; exit 1; }
python3 -m venv --system-site-packages --without-pip .venv-jlens
lens_pip=(.venv/bin/python -m pip --python "$PWD/.venv-jlens/bin/python")
# Use public PyPI only; the container's optional NVIDIA index may not resolve.
export PIP_CONFIG_FILE=/dev/null
export PIP_INDEX_URL=https://pypi.org/simple
export PIP_EXTRA_INDEX_URL=''
# Record the actual inherited module, not a downloadable CUDA wheel constraint.
python3 -c 'import json, torch; from pathlib import Path; print(json.dumps(dict(version=torch.__version__, path=str(Path(torch.__file__).resolve()))))' > .venv-jlens/container-torch.json
"${lens_pip[@]}" install -r requirements-jlens.txt
# All non-torch dependencies are installed above. Never ask pip to resolve the
# container-specific torch build through jlens' unversioned dependency on torch.
"${lens_pip[@]}" install --no-deps -r requirements-jlens-library.txt
"${lens_pip[@]}" install --no-deps -e .
.venv-jlens/bin/python - <<'PY_CHECK'
import json
from pathlib import Path
import jlens
import torch
import transformers
expected = json.loads(Path('.venv-jlens/container-torch.json').read_text())
actual = dict(version=torch.__version__, path=str(Path(torch.__file__).resolve()))
if actual != expected:
    raise RuntimeError(f'Lens environment must reuse container PyTorch: {actual} != {expected}')
print(jlens.__file__, torch.__version__, transformers.__version__)
print('J-lens setup complete; container PyTorch preserved.')
PY_CHECK
"${lens_pip[@]}" freeze > .venv-jlens/environment.txt
