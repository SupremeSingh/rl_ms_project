import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import os
import shutil
import pytest
from math_rl.data import make_row
from math_rl.prompts import MODEL_ID, INSTRUCTION, display_prompt, validate_assets, validate_prompt
from math_rl.train import comparable_config

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("report_resume", ROOT / "scripts/report_resume.py")
report_resume = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report_resume)


def test_prompt_overrides_default_and_review_includes_question():
    row = make_row({"question": "What is 6 * 7?", "answer": "work\n#### 42"}, 3)
    validate_prompt(row["prompt"])
    assert row["prompt"][0] == {"role": "system", "content": INSTRUCTION}
    assert "What is 6 * 7?" in display_prompt(row["prompt"])
    assert "Answer:" in display_prompt(row["prompt"])
    assert row["extra_info"]["prompt_id"] == "gsm8k/train/3"
    with pytest.raises(ValueError):
        validate_prompt(row["prompt"][1:])


def test_wrong_model_and_unpinned_assets_rejected():
    assets = {"model_id": MODEL_ID, "model_revision": "a" * 40, "dataset_revision": "b" * 40}
    validate_assets(assets)
    with pytest.raises(ValueError):
        validate_assets(dict(assets, model_id=MODEL_ID + "-Instruct"))
    with pytest.raises(ValueError):
        validate_assets(dict(assets, model_revision="main"))
    with pytest.raises(ValueError):
        make_row({"question": "fraction", "answer": "#### 1/2"}, 0)


def config():
    return {"trainer": {"total_training_steps": 4, "resume_mode": "disable", "default_local_dir": "a"},
            "actor_rollout_ref": {"actor": {"optim": {"lr": 1e-6}}}, "data": {"seed": 42}}


def test_resume_config_allows_continuation_but_rejects_learning_change():
    original = config()
    resumed = config()
    resumed["trainer"].update(total_training_steps=2, resume_mode="resume_path", default_local_dir="b")
    assert comparable_config(original) == comparable_config(resumed)
    resumed["actor_rollout_ref"]["actor"]["optim"]["lr"] = 1e-3
    assert comparable_config(original) != comparable_config(resumed)
    assert original["trainer"]["total_training_steps"] == 4


def make_run(path, resumed=False):
    path.mkdir()
    record = {"config": config(), "provenance": {"files": {"source": "hash"}}}
    (path / "run.json").write_text(json.dumps(record))
    attempt = path / "attempts/first"
    attempt.mkdir(parents=True)
    if resumed:
        record["config"]["trainer"].update(resume_mode="resume_path", resume_from_path="a/global_step_2")
    (attempt / "run.json").write_text(json.dumps(record))
    (attempt / "console.log").write_text("\n".join(
        f"(TaskRunner pid=123) step:{i} - actor/grad_norm:0.100 - actor/pg_loss:-0.010"
        for i in (range(3, 5) if resumed else range(1, 5))))
    if resumed:
        initial = path / "attempts/initial"
        initial.mkdir()
        (initial / "console.log").write_text("\n".join(
            f"step:{i} - actor/grad_norm:0.100 - actor/pg_loss:-0.010" for i in (1, 2)))
    for step in (2, 4):
        checkpoint = path / f"global_step_{step}"
        (checkpoint / "actor").mkdir(parents=True)
        (checkpoint / "data.pt").touch()
        for name in ("model", "optim", "extra_state"):
            (checkpoint / "actor" / f"{name}_world_size_1_rank_0.pt").touch()
    (path / "rollouts").mkdir()
    for step in range(1, 5):
        rows = [{"input": "question", "output": str(reward), "score": reward, "step": step} for reward in (0, 1)]
        (path / "rollouts" / f"{step}.jsonl").write_text("\n".join(map(json.dumps, rows)))


def test_resume_report_fails_closed(tmp_path):
    control, resumed = tmp_path / "control", tmp_path / "resumed"
    make_run(control)
    make_run(resumed, True)
    assert report_resume.compare(control, resumed)["resume_check_pass"]
    (resumed / "rollouts/3.jsonl").unlink()
    report = report_resume.compare(control, resumed)
    assert not report["resume_check_pass"]
    json.dumps(report, allow_nan=False)


def test_resume_report_detects_prompt_drift_and_missing_optimizer(tmp_path):
    control, resumed = tmp_path / "control", tmp_path / "resumed"
    make_run(control)
    make_run(resumed, True)
    path = resumed / "rollouts/4.jsonl"
    path.write_text(path.read_text().replace("question", "different"))
    (resumed / "global_step_4/actor/optim_world_size_1_rank_0.pt").unlink()
    result = report_resume.compare(control, resumed)
    assert not result["gates"]["prompt_order"]
    assert not result["gates"]["checkpoints_present"]


@pytest.mark.parametrize("valid_checksum", [True, False])
def test_container_wrapper_checks_image_and_preserves_arguments(tmp_path, valid_checksum):
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    for name in ("container_exec.sh", "cluster_env.sh"):
        shutil.copy2(ROOT / "scripts" / name, repo / "scripts" / name)
    runtime = tmp_path / "fake-runtime"
    runtime.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    runtime.chmod(0o755)
    image = tmp_path / "image.sif"
    image.touch()
    (repo / "configs").mkdir()
    digest = hashlib.sha256(image.read_bytes()).hexdigest() if valid_checksum else "0" * 64
    (repo / "configs/container.sha256").write_text(f"{digest}  {image}\n")
    env = dict(os.environ, MATH_RL_ROOT=str(tmp_path), MATH_RL_IMAGE=str(image),
               MATH_RL_RUNTIME=str(runtime), MATH_RL_GPU="1")
    result = subprocess.run(["bash", str(repo / "scripts/container_exec.sh"),
                             "python", "argument with spaces"], env=env, text=True, capture_output=True)
    if not valid_checksum:
        assert result.returncode != 0
        assert "checksum mismatch" in result.stderr
        return
    assert result.returncode == 0, result.stderr
    output = result.stdout
    assert "--nv\n" in output
    assert f"{tmp_path}:{tmp_path}\n" in output
    assert f"--pwd\n{repo}\n" in output
    assert output.endswith("python\nargument with spaces\n")
