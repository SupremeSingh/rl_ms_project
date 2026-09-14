"""VERL batch-reward hook. Keep symbolic dependencies outside the Torch environment."""
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def verifier_command():
    python = ROOT / ".venv-ppo-verifier/bin/python"
    if not python.is_file():
        raise RuntimeError("Run scripts/setup_ppo.sh inside the container first")
    return [str(python), "-m", "math_rl.verifier_batch"]


def compute_score(data_sources, solution_strs, ground_truths, extra_infos=None):
    if not (len(data_sources) == len(solution_strs) == len(ground_truths)):
        raise ValueError("Reward batch lengths differ")
    if any(source != "gsm8k" for source in data_sources):
        raise ValueError("This reward contract has only been checked on numeric GSM8K")
    if not solution_strs:
        return []
    rows = [dict(response=text, ground_truth=gold) for text, gold in zip(solution_strs, ground_truths)]
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    # One child per batch, not per token/answer. Library timeouts remain per answer.
    try:
        result = subprocess.run(verifier_command(), input=json.dumps(rows), text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
                                timeout=15 + 8 * len(rows), cwd=ROOT, env=env)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Isolated verifier failed:\n{exc.stderr[-4000:]}") from exc
    scores = json.loads(result.stdout)
    if len(scores) != len(rows):
        raise RuntimeError("Verifier returned an incomplete batch")
    return scores
