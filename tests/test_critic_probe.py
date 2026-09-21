import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from math_rl.critic_probe import (HEADS, assessment, make_head, metrics, prefix_states,
                                 fit_ridge, fit_trial, predict, sample_shard, transitions)

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("frozen_critics", ROOT / "scripts/frozen_critics.py")
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


def test_exact_architectures_and_linear_value_is_unbounded():
    for kind, count in zip(HEADS, (1, 2, 10, 1)):
        head = make_head(kind, 12, width=16)
        assert sum(isinstance(m, nn.Linear) for m in head.modules()) == count
        assert head(torch.ones(3, 12)).shape == (3, 1)
    head = make_head("linear_value", 2)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.fill_(2)
    assert head(torch.zeros(1, 2)).item() == 2


def test_qwen_features_are_causal_post_norm_and_pre_action():
    from transformers import Qwen2Config, Qwen2Model
    torch.manual_seed(42)
    model = Qwen2Model(Qwen2Config(vocab_size=64, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        attention_dropout=0.)).eval().requires_grad_(False)
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
    captured = []
    hook = model.norm.register_forward_hook(lambda _, args, output: captured.append(output.detach().clone()))
    features = prefix_states(model, tokens, 3, 3)
    hook.remove()
    torch.testing.assert_close(features, captured[0][0, 2:])
    assert features.shape == (4, 16) and not features.requires_grad
    changed = tokens.clone()
    changed[0, 4:] = torch.tensor([17, 18])
    other = prefix_states(model, changed, 3, 3)
    torch.testing.assert_close(features[:2], other[:2])
    with torch.no_grad():
        prompt_only = model(tokens[:, :3]).last_hidden_state[0, -1]
    torch.testing.assert_close(features[0], prompt_only)
    saved = [p.clone() for p in model.parameters()]
    head = make_head("linear_value", 16)
    head(features.clone()).sum().backward()
    assert head.weight.grad is not None
    assert all(p.grad is None and torch.equal(p, before) for p, before in zip(model.parameters(), saved))


def test_transitions_and_random_prefixes_exclude_post_terminal_state():
    states = torch.arange(8.).reshape(4, 2)
    current, following, rewards, done = transitions(states, 1)
    torch.testing.assert_close(current, states[:3])
    torch.testing.assert_close(following[:2], states[1:3])
    assert following[-1].eq(0).all()
    assert rewards.tolist() == [0, 0, 1] and done.tolist() == [False, False, True]
    shard = dict(features=torch.cat([states, torch.ones(9, 2)]), offsets=torch.tensor([0, 4, 13]),
                 rewards=torch.tensor([1, 0]))
    samples = sample_shard(shard, 7)
    assert samples["position"][[0, 8]].tolist() == [0, 0]
    assert samples["position"][1:8].ge(1).all() and samples["position"][:8].lt(3).all()
    assert samples["position"][9:].ge(1).all() and samples["position"][8:].lt(8).all()
    assert samples["y"].tolist() == [1] * 8 + [0] * 8
    assert samples["question"].tolist() == [7] * 16
    repeat = sample_shard(shard, 7)
    torch.testing.assert_close(samples["x"], repeat["x"])
    flipped = sample_shard(dict(shard, rewards=1 - shard["rewards"]), 7)
    torch.testing.assert_close(samples["position"], flipped["position"])
    torch.testing.assert_close(samples["x"][:8], states[samples["position"][:8]])
    single = sample_shard(dict(features=states[:2], offsets=torch.tensor([0, 2]), rewards=torch.tensor([1])), 7)
    assert single["position"].tolist() == [0] * 8


def test_random_prefixes_cover_the_interior_uniformly():
    shard = dict(features=torch.arange(11.).reshape(-1, 1).repeat(1000, 1),
                 offsets=torch.arange(1001) * 11, rewards=torch.zeros(1000))
    sampled = sample_shard(shard, 42)["position"].reshape(-1, 8)
    counts = torch.bincount(sampled[:, 1:].flatten(), minlength=10)[1:]
    assert counts.min() > 650 and counts.max() < 900


def test_probability_metrics_constant_ties_and_calibration():
    y = np.array([0, 1, 0, 1])
    result = metrics(y, np.full(4, .5))
    assert result["auroc"] == .5 and result["accuracy"] == .5
    assert result["brier"] == .25 and result["ece"] == 0
    assert metrics(y, y)["auroc"] == 1
    assert metrics(np.ones(4), np.ones(4))["auroc"] is None
    assert metrics(y, np.array([-1., 2., 0., 1.]))["outside_probability_range"] == .5
    data = dict(y=torch.tensor(y), question=torch.tensor([0, 0, 1, 1]), position=torch.tensor([0, 1, 0, 32]))
    report = assessment(data, y.astype(float), .5)
    assert report["question_mean_brier_difference_vs_constant"] == -.25
    assert report["question_bootstrap_95pct_interval"] == [-.25, -.25]
    assert report["by_prefix"]["question_only"]["n"] == 2


def test_selection_excludes_old_data_and_splits_by_question(tmp_path, monkeypatch):
    eval_spec = importlib.util.spec_from_file_location("evaluate_pilots", ROOT / "scripts/evaluate_pilots.py")
    evaluation = importlib.util.module_from_spec(eval_spec)
    eval_spec.loader.exec_module(evaluation)
    monkeypatch.setitem(sys.modules, "evaluate_pilots", evaluation)

    class Source(list):
        def add_column(self, name, values):
            return Source(dict(row, **{name: v}) for row, v in zip(self, values))
        def shuffle(self, seed):
            return self

    source = Source(dict(question=f"Question {i}?", answer="work #### 4") for i in range(1050))
    development = [{"extra_info": {"prompt_id": "gsm8k/train/1025"},
                    "prompt": [{}, {"content": "Question 1025?"}]}]
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(Dataset=SimpleNamespace(
        from_parquet=lambda _: development), load_dataset=lambda *a, **kw: source))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=SimpleNamespace(
        from_pretrained=lambda *a, **kw: SimpleNamespace(encode=lambda *a, **kw: [1, 2]))))
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    for name in ("configs", "data/gsm8k", "outputs/audit", "outputs/critics-old"):
        (tmp_path / name).mkdir(parents=True)
    pipeline.atomic_json(tmp_path / "configs/assets.json", dict(model_id="Qwen/Qwen2.5-Math-1.5B",
        model_revision="a" * 40, dataset_revision="b" * 40, seed=42))
    pipeline.atomic_json(tmp_path / "data/gsm8k/eval-500.json", {"questions": [
        {"id": "gsm8k/train/1026", "question": "Question 1026?"}]})
    pipeline.atomic_json(tmp_path / "outputs/audit/metadata.json", {"mode": "audit"})
    (tmp_path / "outputs/audit/responses.jsonl").write_text(json.dumps({"prompt_id": "gsm8k/train/1027"}) + "\n")
    pipeline.atomic_json(tmp_path / "outputs/critics-old/questions.json", {"questions": [
        {"id": "gsm8k/train/1028", "question": "Question 1028?", "split": "test"}]})
    rows = pipeline.select_data([2, 1, 1])["questions"]
    assert [r["id"] for r in rows] == [f"gsm8k/train/{i}" for i in (1024, 1029, 1030, 1031)]
    assert [r["split"] for r in rows] == ["train", "train", "val", "test"]


def test_generation_keeps_base_sampling_and_resumes_without_replacing_data(tmp_path, monkeypatch):
    import math_rl.ppo_reward as reward
    (tmp_path / "trajectories").mkdir()
    calls = []

    class Engine:
        def __init__(self, **kwargs):
            assert kwargs["model"].endswith("models/qwen-math")
        def generate(self, prompts, params):
            assert prompts == [{"prompt_token_ids": [1, 2]}]
            assert (params["temperature"], params["top_p"], params["max_tokens"], params["seed"]) == (1, 1, 2048, 42)
            calls.append(params)
            return [SimpleNamespace(outputs=[SimpleNamespace(text="Answer 4", token_ids=[4, 5],
                finish_reason="stop", stop_reason=5) for _ in range(params["n"])])]

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=Engine, SamplingParams=lambda **kw: kw))
    def score(sources, responses, gold, **kwargs):
        assert kwargs['exclude_errors'] is True
        assert sources == ["gsm8k"] * 2 and gold == ["4"] * 2
        return [dict(score=1, verifier_status="correct", extracted="4")] * 2
    questions = [dict(id="q1", prompt_token_ids=[1, 2], ground_truth="4")]
    def failed_score(*args, **kwargs):
        raise RuntimeError("scorer subprocess failed")
    monkeypatch.setattr(reward, "compute_score", failed_score)
    with pytest.raises(RuntimeError, match="scorer subprocess"):
        pipeline.generate(tmp_path, dict(responses=2), questions)
    assert (tmp_path / "pending/00000.json").exists()
    assert not pipeline.shard_path(tmp_path, 0, "trajectories").exists()
    monkeypatch.setattr(reward, "compute_score", score)
    pipeline.generate(tmp_path, dict(responses=2), questions)
    assert not (tmp_path / "pending/00000.json").exists()
    path = pipeline.shard_path(tmp_path, 0, "trajectories")
    frozen = path.read_bytes()
    assert json.loads(frozen)["responses"][0]["token_ids"] == [4, 5]
    pipeline.generate(tmp_path, dict(responses=2), questions)
    assert path.read_bytes() == frozen and len(calls) == 1


def test_fitting_all_four_heads_from_saved_features(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    for name in ("features", "trajectories", "heads"):
        (tmp_path / name).mkdir()
    questions = []
    for i in range(12):
        q = dict(id=f"q{i}", split=("train", "val", "test")[i // 4], question=f"Question {i}?", ground_truth="4")
        questions.append(q)
        path = pipeline.shard_path(tmp_path, i, "trajectories")
        pipeline.atomic_json(path, dict(responses=[dict(score=r, finish_reason="stop", response="Example", token_ids=[1, 2, 3, 4],
            verifier_status="verify_timeout" if i == 0 and r == 0 else ("correct" if r else "incorrect")) for r in (0, 1)]))
        features = torch.randn(10, 8) * .05
        features[:5, 0], features[5:, 0] = -1, 1
        pipeline.save_tensor(pipeline.shard_path(tmp_path, i, "features"), dict(features=features[5:] if i == 0 else features,
            offsets=torch.tensor([0, 5] if i == 0 else [0, 5, 10]), rewards=torch.tensor([1] if i == 0 else [0, 1]), question_id=q["id"],
            response_indices=torch.tensor([1] if i == 0 else [0, 1]), exclusion_policy=pipeline.EXCLUSION_POLICY,
            trajectories_sha256=pipeline.sha256(path)))
    config = dict(seeds=[42], epochs=2, min_epochs=1, patience=2, learning_rates=[1e-3, 1e-4])
    pipeline.fit(tmp_path, config, questions)
    report = json.loads((tmp_path / "summary.json").read_text())
    assert report["trajectories"] == 24 and report["reward_rate"] == pytest.approx(12 / 23)
    assert report['retained_trajectories'] == 23 and report['excluded_trajectories'] == 1
    assert report["verifier_timeouts"] == 1 and report["verifier_timeout_rate"] == pytest.approx(1 / 24)
    assert len(json.loads((tmp_path / "verifier-timeouts.json").read_text())) == 1
    assert {r["kind"] for r in report["heads"]} == set(HEADS)
    assert report["prefix_counts"] == dict(train=56, val=64, test=64)
    assert report["ridge"]["test"]["brier"] < report["constant"]["brier"]
    assert report["ridge"]["equation_residual"] < 1e-10
    for row in report["heads"]:
        assert np.isfinite(row["test"]["brier"])
        assert len(row["fitting_curves"]) == 2
        assert (tmp_path / "heads" / f"{row['kind']}-42-predictions.npy").exists()
    assert set(torch.load(tmp_path / "train.pt", weights_only=True)["question"].tolist()).isdisjoint(
        torch.load(tmp_path / "test.pt", weights_only=True)["question"].tolist())
    previous = (tmp_path / "summary.json").read_bytes()
    pipeline.fit(tmp_path, config, questions)
    assert (tmp_path / "summary.json").read_bytes() == previous


def test_feature_extraction_excludes_failures_and_handles_empty_question(tmp_path, monkeypatch):
    from math_rl import critic_probe
    for name in ('features', 'trajectories'):
        (tmp_path / name).mkdir()
    class Backbone:
        config = SimpleNamespace(hidden_size=2)
        def to(self, *args): return self
        def eval(self): return self
        def requires_grad_(self, *args): return self
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(AutoModel=SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: Backbone())))
    original_tensor = torch.tensor
    def cpu_tensor(*args, **kwargs):
        kwargs.pop('device', None)
        return original_tensor(*args, **kwargs)
    monkeypatch.setattr(torch, 'tensor', cpu_tensor)
    calls = []
    def states(model, tokens, prompt_length, response_length):
        calls.append(tokens.tolist())
        return torch.ones(response_length + 1, 2)
    monkeypatch.setattr(critic_probe, 'prefix_states', states)
    bad = dict(response='bad', token_ids=[3], score=None, verifier_status='verify_error')
    good = dict(response='good', token_ids=[4, 5], score=1, verifier_status='correct')
    questions = [dict(id=f'q{i}', prompt_token_ids=[1, 2]) for i in range(2)]
    for i, rows in enumerate(([bad, good], [bad, bad])):
        pipeline.atomic_json(pipeline.shard_path(tmp_path, i, 'trajectories'), dict(question_id=f'q{i}', responses=rows))
    pipeline.extract(tmp_path, dict(responses=2), questions)
    assert calls == [[[1, 2, 4, 5]]]
    kept = torch.load(pipeline.shard_path(tmp_path, 0, 'features'), weights_only=True)
    assert kept['response_indices'].tolist() == [1] and kept['rewards'].tolist() == [1]
    empty = torch.load(pipeline.shard_path(tmp_path, 1, 'features'), weights_only=True)
    assert empty['features'].shape == (0, 2) and len(empty['rewards']) == 0
    with pytest.raises(ValueError, match='No usable answers'):
        pipeline.prepare_probes(tmp_path, [dict(q, split='train') for q in questions])


def test_linear_fit_has_prior_checkpoint_and_stops_on_validation():
    x = torch.zeros(64, 2)
    y = torch.tensor([0., 1., 1., 1.] * 16)
    data = dict(x=x, y=y)
    mean, scale = torch.zeros(2), torch.ones(2)
    for kind in HEADS:
        fit = fit_trial(kind, data, data, mean, scale, torch.device("cpu"), 42, .001,
                        epochs=100, width=4, min_epochs=4, patience=3)
        assert fit["curve"][0]["validation_brier"] == pytest.approx(.1875)
        assert fit["best_epoch"] == 0 and fit["epochs_run"] == 4 and fit["stopped_early"]
        head = make_head(kind, 2, width=4)
        head.load_state_dict(fit["state"])
        np.testing.assert_allclose(predict(head, kind, data, mean, scale, torch.device("cpu")), .75)


def test_ridge_matches_regularized_equations_and_preserves_bias():
    torch.manual_seed(1)
    x = torch.randn(100, 3)
    # Deliberately nonzero offset to test unpenalized intercept.
    y = .7 + x @ torch.tensor([.1, -.2, .3])
    data = dict(x=x, y=y)
    mean, scale = torch.zeros(3), torch.ones(3)
    fit = fit_ridge(data, data, mean, scale, alphas=(.2,))
    design = torch.cat([x.double(), torch.ones(100, 1)], dim=1)
    penalty = torch.diag(torch.tensor([.2, .2, .2, 0.], dtype=torch.float64))
    expected = torch.linalg.solve(design.T @ design / 100 + penalty, design.T @ y.double() / 100)
    actual = torch.cat([fit["state"]["weight"].flatten(), fit["state"]["bias"]])
    torch.testing.assert_close(actual.double(), expected, atol=1e-7, rtol=1e-7)
    assert fit["equation_residual"] < 1e-12
    zero = dict(x=torch.zeros_like(x), y=torch.full_like(y, .7))
    constant = fit_ridge(zero, zero, mean, scale, alphas=(1.,))
    assert constant["state"]["bias"].item() == pytest.approx(.7)
