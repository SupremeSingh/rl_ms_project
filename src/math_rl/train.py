"""Launch pinned VERL and retain evidence for smoke/resume comparisons."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from math_rl.provenance import snapshot, write_json


def comparable_config(value):
    value = json.loads(json.dumps(value))
    trainer = value["trainer"]
    for key in ("default_local_dir", "resume_mode", "resume_from_path", "total_training_steps",
                "rollout_data_dir", "experiment_name", "validation_data_dir"):
        trainer.pop(key, None)
    value["actor_rollout_ref"]["actor"]["optim"].pop("total_training_steps", None)
    return value


def main():
    import yaml
    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env.setdefault("VLLM_USE_V1", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    command = [sys.executable, "-m", "verl.trainer.main_ppo",
               "--config-dir", str(root / "configs"), "--config-name", "smoke", *sys.argv[1:]]
    if any(arg in ("--cfg", "--help", "-h") or arg.startswith("--cfg=") for arg in sys.argv[1:]):
        subprocess.run(command, cwd=root, env=env, check=True)
        return
    subprocess.run([sys.executable, str(root / "scripts/preflight.py")], cwd=root, env=env, check=True)
    resolved = subprocess.check_output(command + ["--cfg", "job", "--resolve"], cwd=root, env=env, text=True)
    config = yaml.safe_load(resolved)
    output = Path(config["trainer"]["default_local_dir"])
    if not output.is_absolute():
        output = root / output
    provenance = snapshot(root)
    mode = config["trainer"]["resume_mode"]
    if mode == "disable" and output.exists() and any(output.iterdir()):
        raise ValueError("Fresh run directory is nonempty; choose a new trainer.default_local_dir")
    if mode == "auto":
        raise ValueError("Use explicit resume_path for an auditable smoke test")
    if mode == "resume_path":
        checkpoint = Path(config["trainer"]["resume_from_path"])
        if not checkpoint.is_absolute():
            checkpoint = root / checkpoint
        if not (checkpoint / "data.pt").is_file() or not (checkpoint / "actor").is_dir():
            raise ValueError("Incomplete resume checkpoint")
        previous = json.loads((checkpoint.parent / "run.json").read_text())
        if comparable_config(previous["config"]) != comparable_config(config):
            raise ValueError("Resume configuration differs from original run")
        if previous["provenance"]["files"] != provenance["files"]:
            raise ValueError("Code, data, assets, or runtime manifest changed since original run")
    output.mkdir(parents=True, exist_ok=True)
    record = {"config": config, "provenance": provenance}
    if not (output / "run.json").exists():
        write_json(output / "run.json", record)
    attempt = output / "attempts" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    attempt.mkdir(parents=True)
    write_json(attempt / "run.json", record)
    (attempt / "resolved.yaml").write_text(resolved)
    with (attempt / "console.log").open("w") as log:
        process = subprocess.Popen(command, cwd=root, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
            # Run-level evidence is saved BEFORE training, even if the job is killed.
            for checkpoint in output.glob("global_step_*"):
                if checkpoint.is_dir() and (checkpoint / "data.pt").exists():
                    target = checkpoint / "run.json"
                    if not target.exists():
                        shutil.copy2(attempt / "run.json", target)
    if code:
        raise subprocess.CalledProcessError(code, command)


if __name__ == "__main__":
    main()
