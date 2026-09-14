"""Exercise the batch orchestration with fake Slurm and training processes."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("failure", ["none", "setup", "ppo", "report"])
def test_parallel_pilots_and_failure_reporting(tmp_path, failure):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy2(ROOT / "scripts/submit_pilots.sh", scripts)
    (scripts / "cluster_env.sh").write_text(":\n")
    (scripts / "container_exec.sh").write_text('exec "$TEST_PYTHON" fake_runner.py "$@"\n')
    (tmp_path / "fake_runner.py").write_text('''
import json, os, sys, time
from pathlib import Path
failure = os.environ["TEST_FAILURE"]
if "scripts/setup_ppo.sh" in sys.argv:
    assert os.environ["MATH_RL_GPU"] == "0"
    if failure == "setup":
        sys.exit(3)
    Path("setup-done").touch()
    sys.exit(0)
assert Path("setup-done").exists()
method = sys.argv[sys.argv.index("--config-name") + 1]
assert os.environ["MATH_RL_GPU"] == "1"
assert os.environ["RAY_ADDRESS"] == "local"
assert f"trainer.default_local_dir=checkpoints/{method}-{os.environ['SLURM_JOB_ID']}" in sys.argv
assert f"hydra.run.dir=outputs/pilots-{os.environ['SLURM_JOB_ID']}/{method}-hydra" in sys.argv
Path(f"{method}-started").write_text(json.dumps({"ray_tmp": os.environ["RAY_TMPDIR"]}))
# Neither trainer can finish unless the other was launched concurrently.
deadline = time.monotonic() + 8
while not all(Path(f"{name}-started").exists() for name in ("ppo", "grpo")):
    if time.monotonic() > deadline:
        sys.exit(9)
    time.sleep(0.02)
Path(f"{method}-finished").touch()
sys.exit(7 if method == failure else 0)
''')
    (scripts / "report_ppo.py").write_text('''
import os, sys
from pathlib import Path
assert Path("ppo-finished").exists() and Path("grpo-finished").exists()
method = Path(sys.argv[1]).name.split("-")[0]
Path(f"{method}-reported").touch()
sys.exit(2 if method == "ppo" and os.environ["TEST_FAILURE"] == "report" else 0)
''')
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    srun = bin_dir / "srun"
    srun.write_text(f"#!{sys.executable}\n" + '''
import os, sys
args = sys.argv[1:]
expected = ["--exclusive", "--exact", "--nodes=1", "--ntasks=1",
            "--cpus-per-task=8", "--gres=gpu:a5000:2", "--mem=128G"]
assert args[:len(expected)] == expected, args
args = args[len(expected):]
os.execvp(args[0], args)
''')
    srun.chmod(0o755)
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}",
               TEST_PYTHON=sys.executable, TEST_FAILURE=failure,
               SLURM_SUBMIT_DIR=str(tmp_path), SLURM_JOB_ID=str(os.getpid()))
    result = subprocess.run(["bash", str(scripts / "submit_pilots.sh")], env=env,
                            capture_output=True, text=True, timeout=30)
    run = tmp_path / f"outputs/pilots-{os.getpid()}"
    status = dict(line.split("=", 1) for line in (run / "status.txt").read_text().splitlines())
    assert result.returncode == (0 if failure == "none" else 1), result.stderr
    if failure == "setup":
        assert status["setup_exit"] == "3"
        assert status["ppo_exit"] == status["grpo_exit"] == "not_started"
        assert not (tmp_path / "ppo-started").exists()
        assert not (tmp_path / "grpo-started").exists()
        return
    assert status == {"setup_exit": "0", "ppo_exit": "7" if failure == "ppo" else "0",
                      "grpo_exit": "0", "ppo_report_exit": "2" if failure == "report" else "0",
                      "grpo_report_exit": "0"}
    paths = [json.loads((tmp_path / f"{name}-started").read_text())["ray_tmp"]
             for name in ("ppo", "grpo")]
    assert paths[0] != paths[1]
    for name in ("ppo", "grpo"):
        assert (tmp_path / f"{name}-reported").exists()
        assert (run / f"{name}.log").exists()
        assert (run / f"{name}-report.log").exists()
