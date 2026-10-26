"""Small, CPU-only provenance records; no credentials or full environment dump."""
import os
from pathlib import Path
import subprocess
from functools import partial
from slice_critic.storage import sha256, write_json as _write_json

write_json = partial(_write_json, sort_keys=True)


def snapshot(root):
    root = Path(root)
    files = {}
    for pattern in ("src/**/*.py", "packages/slice-critic/src/**/*.py", "packages/slice-critic/pyproject.toml", "scripts/*.py", "scripts/*.sh", "configs/*.yaml", "pyproject.toml", "requirements*.txt"):
        for path in sorted(root.glob(pattern)):
            files[str(path.relative_to(root))] = sha256(path)
    for name in ("configs/assets.json", "configs/environment.txt", "configs/container.sha256",
                 "configs/ppo-verifier-environment.txt"):
        path = root / name
        if path.exists():
            files[name] = sha256(path)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"git_commit": commit, "files": files,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "container_path": os.environ.get("MATH_RL_IMAGE")}
