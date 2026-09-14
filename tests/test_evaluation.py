"""Evaluation correctness without downloading models or allocating GPUs."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("evaluate_pilots", ROOT / "scripts/evaluate_pilots.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class Tokenizer:
    def encode(self, text, add_special_tokens):
        assert not add_special_tokens
        assert "####" not in text
        return [1] * (513 if "OVERLONG" in text else 20)


def test_selection_excludes_development_duplicates_and_long_prompts():
    source = [dict(original_index=i, question=q, answer="work #### 1234")
              for i, q in enumerate(["Training?", "Audit?", "Audit?", "Duplicate?",
                                     "OVERLONG?", "New?", "New?", "Another?"])]
    rows = evaluation.select_questions(source, Tokenizer(), 2,
                                       {"gsm8k/train/0", "gsm8k/train/1"}, {"Audit?", "Duplicate?"})
    assert [r["id"] for r in rows] == ["gsm8k/train/5", "gsm8k/train/7"]
    assert all("1234" not in r["prompt"] for r in rows)
    evaluation.validate_questions(rows, 2)
    with pytest.raises(ValueError, match="overlaps"):
        evaluation.validate_questions(rows, 2, excluded_questions={"New?"})
    with pytest.raises(ValueError, match="Duplicate evaluation questions"):
        evaluation.validate_questions([rows[0], dict(rows[1], question=" New? ")], 2)
    with pytest.raises(ValueError, match="need 3"):
        evaluation.select_questions(source, Tokenizer(), 3, {r["id"] for r in rows},
                                    {"Training?", "Audit?", "Duplicate?"})


def test_question_manifest_is_frozen_and_excludes_saved_audits(tmp_path, monkeypatch):
    class Source(list):
        def add_column(self, name, values):
            return Source(dict(row, **{name: value}) for row, value in zip(self, values))

        def shuffle(self, seed):
            assert seed == 42
            return self  # A deterministic source; production uses Dataset.shuffle.

    source = Source(dict(question=q, answer="work #### 4") for q in
                    ["Reserved 0?", "Reserved 1?", "Training?", "Audit?", " Training? ",
                     "OVERLONG?", "New?", "Another?"])
    development = [{"extra_info": {"prompt_id": "gsm8k/train/2"},
                    "prompt": [{}, {"content": "Training?"}]}]
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(
        Dataset=SimpleNamespace(from_parquet=lambda _: development), load_dataset=lambda *a, **kw: source))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: Tokenizer())))
    monkeypatch.setattr(evaluation, "OFFSET", 2)
    for folder in ("configs", "src/math_rl", "data/gsm8k", "outputs/audit"):
        (tmp_path / folder).mkdir(parents=True)
    evaluation.write_json(tmp_path / "configs/assets.json", dict(model_id="Qwen/Qwen2.5-Math-1.5B",
        model_revision="a" * 40, dataset_revision="b" * 40, seed=42))
    (tmp_path / "src/math_rl/prompts.py").write_text("prompt code")
    evaluation.write_json(tmp_path / "outputs/audit/metadata.json", {"mode": "audit"})
    evaluation.write_jsonl(tmp_path / "outputs/audit/responses.jsonl", [{"prompt_id": "gsm8k/train/3"}])
    path = evaluation.prepare_questions(tmp_path, 2)
    frozen = path.read_bytes()
    assert [q["id"] for q in json.loads(frozen)["questions"]] == ["gsm8k/train/6", "gsm8k/train/7"]
    assert evaluation.prepare_questions(tmp_path, 2) == path
    assert path.read_bytes() == frozen
    source[-1]["answer"] = "work #### 5"
    with pytest.raises(ValueError, match="Frozen questions"):
        evaluation.prepare_questions(tmp_path, 2)
    assert path.read_bytes() == frozen


def make_results(out):
    questions = [dict(id=f"q{i}", question=f"Question {i}?", ground_truth="4",
                      prompt_token_ids=[i], prompt=f"Question {i}? Solution:") for i in range(4)]
    evaluation.write_json(out / "questions.json", {"questions": questions})
    for label, scores in dict(base=[0, 0, 1, 1], ppo=[1, 1, 0, 1], grpo=[0, 0, 1, 1]).items():
        folder = out / label
        folder.mkdir()
        rows = [dict(id=q["id"], ground_truth="4", score=score, extracted="4" if score else "5",
                     response="The answer is 4." if score else "The answer is 5.", response_tokens=8,
                     finish_reason="stop", verifier_status="correct" if score else "incorrect")
                for q, score in zip(questions, scores)]
        evaluation.write_jsonl(folder / "responses.jsonl", rows)
        evaluation.write_json(folder / "metadata.json", dict(
            questions_sha256=evaluation.sha256(out / "questions.json"), sampling=evaluation.SETTINGS,
            responses_sha256=evaluation.sha256(folder / "responses.jsonl"),
            gpu="fake GPU", torch="test", vllm="test", transformers="test"))
    return questions


def test_paired_report_and_review_artifacts(tmp_path):
    make_results(tmp_path)
    evaluation.report(tmp_path)
    result = json.loads((tmp_path / "summary.json").read_text())
    assert result["models"]["base"]["accuracy"] == .5
    assert result["models"]["ppo"]["accuracy"] == .75
    ppo = result["versus_base"]["ppo"]
    assert (ppo["wrong_to_right"], ppo["right_to_wrong"], ppo["net_correct"]) == (2, 1, 1)
    assert ppo["accuracy_change_percentage_points"] == 25
    assert result["versus_base"]["grpo"]["paired_p_value"] == 1
    assert len(evaluation.read_jsonl(tmp_path / "changes.jsonl")) == 3
    review = (tmp_path / "review.txt").read_text()
    assert all(review.count(label.upper() + " |") == 4 for label in evaluation.LABELS)
    assert evaluation.paired_counts([0] * 6, [1] * 6)["paired_p_value"] == .03125


@pytest.mark.parametrize("problem", ["missing", "duplicate", "reference", "runtime", "status", "sampling", "tampered"])
def test_report_refuses_invalid_comparisons(tmp_path, problem):
    make_results(tmp_path)
    folder = tmp_path / "ppo"
    rows = evaluation.read_jsonl(folder / "responses.jsonl")
    metadata = json.loads((folder / "metadata.json").read_text())
    if problem == "missing":
        rows.pop()
    elif problem == "duplicate":
        rows[1]["id"] = rows[0]["id"]
    elif problem == "reference":
        rows[0]["ground_truth"] = "42"
    elif problem == "runtime":
        rows[0]["verifier_status"] = "parse_timeout"
    elif problem == "status":
        rows[0]["score"] = 0
    elif problem == "sampling":
        metadata["sampling"]["temperature"] = 1
    elif problem == "tampered":
        rows[0]["response"] = "Changed after scoring"
    (folder / "responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    if problem != "tampered":
        metadata["responses_sha256"] = evaluation.sha256(folder / "responses.jsonl")
    evaluation.write_json(folder / "metadata.json", metadata)
    with pytest.raises(ValueError):
        evaluation.report(tmp_path)
    assert not (tmp_path / "summary.json").exists()


def test_checkpoint_requires_matching_provenance_and_every_actor_shard(tmp_path):
    for name in evaluation.TRAINING_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original")
    run = tmp_path / "checkpoints/ppo-123"
    actor = run / "global_step_10/actor"
    (actor / "huggingface").mkdir(parents=True)
    (actor / "huggingface/config.json").write_text("{}")
    (actor.parent / "data.pt").write_bytes(b"data")
    evaluation.write_json(actor / "fsdp_config.json", {"world_size": 2})
    record = dict(config=dict(algorithm={"adv_estimator": "gae"},
        data={"train_files": "data/gsm8k/train.parquet", "val_files": "data/gsm8k/val.parquet"},
        trainer={"n_gpus_per_node": 2, "nnodes": 1}), provenance=dict(verifier={},
        files={name: evaluation.sha256(tmp_path / name) for name in evaluation.TRAINING_FILES}))
    evaluation.write_json(run / "run.json", record)
    for rank in range(2):
        (actor / f"model_world_size_2_rank_{rank}.pt").write_bytes(b"weights")
    assert evaluation.inspect_checkpoint(tmp_path, "ppo", 123, 10)[0] == actor
    (actor / "model_world_size_2_rank_1.pt").unlink()
    with pytest.raises(ValueError, match="shards"):
        evaluation.inspect_checkpoint(tmp_path, "ppo", 123, 10)
    (tmp_path / "src/math_rl/prompts.py").write_text("changed")
    with pytest.raises(ValueError, match="provenance"):
        evaluation.inspect_checkpoint(tmp_path, "ppo", 123, 10)


def test_export_checks_merged_weights_and_propagates_failure(tmp_path, monkeypatch):
    calls = []
    model = tmp_path / "model"

    def fake_run(command, **kwargs):
        calls.append(command)
        assert kwargs["check"]
        if "merge" in command:
            model.mkdir()
            (model / "model.safetensors").write_bytes(b"weights")
        else:
            assert command[-2:] == ["--test_hf_dir", str(model)]
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(evaluation.subprocess, "run", fake_run)
    with pytest.raises(subprocess.CalledProcessError):
        evaluation.export_actor(tmp_path / "actor", model, tmp_path / "export.log")
    assert [c[3] for c in calls] == ["merge", "test"]


def test_generation_uses_only_frozen_prompt_tokens_and_shared_verifier(tmp_path, monkeypatch):
    import math_rl.ppo_reward as reward
    questions = make_results(tmp_path)
    captured = {}

    class Engine:
        def __init__(self, **kwargs):
            captured["engine"] = kwargs

        def generate(self, prompts, params):
            assert prompts == [{"prompt_token_ids": q["prompt_token_ids"]} for q in questions]
            assert params == evaluation.SETTINGS
            return [SimpleNamespace(outputs=[SimpleNamespace(text=f"Answer {i}", token_ids=[1, 2],
                                    finish_reason="stop")]) for i in range(len(prompts))]

    def score(sources, answers, references):
        assert sources == ["gsm8k"] * 4 and references == ["4"] * 4
        assert answers == [f"Answer {i}" for i in range(4)]
        return [dict(score=0, extracted="", verifier_status="parse_failure")] * 4

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=Engine, SamplingParams=lambda **kw: kw))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__="test",
                        cuda=SimpleNamespace(get_device_name=lambda: "fake GPU")))
    monkeypatch.setattr(evaluation, "version", lambda _: "test")
    monkeypatch.setattr(reward, "compute_score", score)
    out = tmp_path / "generated"
    evaluation.generate(tmp_path / "trained-model", tmp_path / "questions.json", out)
    assert captured["engine"]["model"] == str(tmp_path / "trained-model")
    assert captured["engine"]["generation_config"] == "vllm"
    assert captured["engine"]["tensor_parallel_size"] == 1
    assert [r["id"] for r in evaluation.read_jsonl(out / "responses.jsonl")] == [q["id"] for q in questions]
    assert json.loads((out / "metadata.json").read_text())["sampling"]["temperature"] == 0


def test_full_evaluation_orchestration_reuses_saved_actors(tmp_path, monkeypatch):
    import math_rl.ppo_reward as reward
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    make_results(fixtures)
    base = tmp_path / "models/qwen-math"
    base.mkdir(parents=True)
    (base / "model.safetensors").write_bytes(b"base")
    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    monkeypatch.setattr(evaluation, "snapshot", lambda _: {})
    monkeypatch.setattr(evaluation, "distribution", lambda _: SimpleNamespace(
        read_text=lambda _: json.dumps({"vcs_info": {"commit_id": evaluation.VERL_COMMIT}})))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: True, device_count=lambda: 1, is_bf16_supported=lambda: True)))
    monkeypatch.setattr(reward, "verifier_command", lambda: ["verifier"])
    monkeypatch.setattr(evaluation.subprocess, "check_output", lambda *a, **kw: "{}")
    monkeypatch.setattr(evaluation, "prepare_questions", lambda *a: fixtures / "questions.json")
    inspected, exported, generated = [], [], []

    def inspect(root, method, job, step):
        inspected.append((method, job, step))
        return tmp_path / f"{method}-actor", {"verifier": {}}

    def export(actor, model, log):
        exported.append(actor.name)
        model.mkdir()
        (model / "model.safetensors").write_bytes(actor.name.encode())

    def generate(command, **kwargs):
        assert kwargs["check"]
        assert command[2] == "generate"
        out = Path(command[command.index("--out") + 1])
        model = Path(command[command.index("--model") + 1])
        generated.append(out.name)
        assert model == (base if out.name == "base" else out.parent / f"{out.name}-model")
        out.mkdir()
        for filename in ("metadata.json", "responses.jsonl"):
            (out / filename).write_bytes((fixtures / out.name / filename).read_bytes())

    monkeypatch.setattr(evaluation, "inspect_checkpoint", inspect)
    monkeypatch.setattr(evaluation, "export_actor", export)
    monkeypatch.setattr(evaluation.subprocess, "run", generate)
    out = tmp_path / "evaluation"
    evaluation.run(SimpleNamespace(out=out, pilot_job=12589388, step=10, count=4))
    assert inspected == [("ppo", 12589388, 10), ("grpo", 12589388, 10)]
    assert exported == ["ppo-actor", "grpo-actor"]
    assert generated == ["base", "ppo", "grpo"]
    assert json.loads((out / "run.json").read_text())["phase"] == "complete"
    assert (out / "summary.json").exists()
