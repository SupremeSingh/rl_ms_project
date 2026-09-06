import os
from pathlib import Path
import subprocess
import sys


def main():
    # This editable-install launcher locates the checkout, wherever called from.
    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env.setdefault("VLLM_USE_V1", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    subprocess.run(
        [sys.executable, "-m", "verl.trainer.main_ppo",
         "--config-dir", str(root / "configs"), "--config-name", "smoke",
         *sys.argv[1:]],
        cwd=root, env=env, check=True,
    )


if __name__ == "__main__":
    main()
