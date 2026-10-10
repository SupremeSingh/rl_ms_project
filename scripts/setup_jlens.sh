#!/usr/bin/env bash
# Run inside our pinned container, not inside the PPO virtual environment.
set -euo pipefail
# Debian's container Python lacks ensurepip. Create the environment without it,
# then use the existing PPO environment's pip solely as an installer targeting
# the NEW environment (the same mechanism as setup_ppo.sh).
[[ -x .venv/bin/python ]] || { echo 'Existing .venv/bin/python is required; run the project setup first.' >&2; exit 1; }
python3 -m venv --system-site-packages --without-pip .venv-jlens
lens_pip=(.venv/bin/python -m pip --python "$PWD/.venv-jlens/bin/python")
# Keep the container's CUDA/PyTorch stack. The lens itself has no torch version floor.
python3 -c 'import torch; print("torch==" + torch.__version__)' > .venv-jlens/torch-constraint.txt
"${lens_pip[@]}" install -c .venv-jlens/torch-constraint.txt -r requirements-jlens.txt
"${lens_pip[@]}" install --no-deps -e .
"${lens_pip[@]}" freeze > .venv-jlens/environment.txt
.venv-jlens/bin/python -c 'import jlens, torch, transformers; print(jlens.__file__, torch.__version__, transformers.__version__)'
