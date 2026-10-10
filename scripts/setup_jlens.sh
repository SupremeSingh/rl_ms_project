#!/usr/bin/env bash
# Run inside our pinned container, not inside the PPO virtual environment.
set -euo pipefail
python3 -m venv --system-site-packages .venv-jlens
# Keep the container's CUDA/PyTorch stack. The lens itself has no torch version floor.
python3 -c 'import torch; print("torch==" + torch.__version__)' > .venv-jlens/torch-constraint.txt
.venv-jlens/bin/python -m pip install -c .venv-jlens/torch-constraint.txt -r requirements-jlens.txt
.venv-jlens/bin/python -m pip install --no-deps -e .
.venv-jlens/bin/python -m pip freeze > .venv-jlens/environment.txt
.venv-jlens/bin/python -c 'import jlens, torch, transformers; print(jlens.__file__, torch.__version__, transformers.__version__)'
