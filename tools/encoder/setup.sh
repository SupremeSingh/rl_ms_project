#!/usr/bin/env bash
# Run INSIDE the existing container; never modify .venv or .venv-critic.
set -euo pipefail
[[ -x .venv/bin/python ]] || { echo 'Existing project .venv is required.'; exit 1; }
[[ ! -e .venv-encoder ]] || { echo '.venv-encoder already exists; inspect it rather than overwriting.'; exit 1; }
python3 -m venv --system-site-packages --without-pip .venv-encoder
python3 -c 'import torch,json; from pathlib import Path; print(json.dumps(dict(version=torch.__version__,path=str(Path(torch.__file__).resolve()))))' > .venv-encoder/container-torch.json
installer=(.venv/bin/python -m pip --python "$PWD/.venv-encoder/bin/python")
export PIP_CONFIG_FILE=/dev/null PIP_INDEX_URL=https://pypi.org/simple PIP_EXTRA_INDEX_URL=''
"${installer[@]}" install -r requirements-encoder.txt
.venv-encoder/bin/python - <<'PY'
import json
from pathlib import Path
import torch,transformers,peft,llm2vec
expected=json.loads(Path('.venv-encoder/container-torch.json').read_text())
assert expected==dict(version=torch.__version__,path=str(Path(torch.__file__).resolve())), 'Container torch changed'
assert transformers.__version__=='4.44.2' and peft.__version__=='0.13.2'
print('Encoder imports passed. Other inherited packages may have incompatible requirements; do not use VERL/vLLM in this environment.')
PY
"${installer[@]}" freeze > .venv-encoder/environment.txt
echo 'Separate encoder environment ready; PPO environments unchanged.'
