"""CPU integration tests; run in the training container to include pinned VERL."""
import os
import json
from pathlib import Path
import pickle
import subprocess
import sys
from types import SimpleNamespace

import pytest

from math_rl.ppo_checks import checkpoint_complete
from math_rl import ppo_reward

ROOT = Path(__file__).resolve().parents[1]


def test_checkpoint_requires_both_models_and_every_rank(tmp_path):
    (tmp_path / "data.pt").write_bytes(b"state")
    for role in ("actor", "critic"):
        (tmp_path / role).mkdir()
        for rank in range(2):
            for kind in ("model", "optim", "extra_state"):
                (tmp_path / role / f"{kind}_world_size_2_rank_{rank}.pt").write_bytes(b"state")
    assert checkpoint_complete(tmp_path, 2)
    (tmp_path / "critic/optim_world_size_2_rank_1.pt").unlink()
    assert not checkpoint_complete(tmp_path, 2)


def test_batch_bridge_uses_separate_interpreter_and_preserves_order(monkeypatch):
    verifier = os.environ.get("MATH_RL_TEST_VERIFIER")
    if not verifier:
        pytest.skip("Set MATH_RL_TEST_VERIFIER to an isolated Math-Verify Python")
    monkeypatch.setattr(ppo_reward, "verifier_command", lambda: [verifier, "-m", "math_rl.verifier_batch"])
    scores = ppo_reward.compute_score(["gsm8k"] * 3,
        [r"The answer is \boxed{4}.", "The answer is 5.", "I do not know."], ["4"] * 3)
    assert [row["score"] for row in scores] == [1, 0, 0]
    assert scores[2]["verifier_status"] == "parse_failure"
    with pytest.raises(RuntimeError, match="input_too_long"):
        ppo_reward.compute_score(["gsm8k"], ["x" * 32001], ["4"])


def test_bridge_rejects_bad_batch():
    with pytest.raises(ValueError):
        ppo_reward.compute_score(["gsm8k"], [], [])
    with pytest.raises(ValueError):
        ppo_reward.compute_score(["math"], ["4"], ["4"])


def test_completion_dataset_uses_exact_tokens_and_can_resume(tmp_path):
    torch = pytest.importorskip("torch")
    datasets = pytest.importorskip("datasets")
    from math_rl.completion_dataset import CompletionDataset
    from math_rl.data import make_row
    from math_rl.prompts import encode_completion
    from torchdata.stateful_dataloader import StatefulDataLoader

    class Tokenizer:
        pad_token_id = 0
        def encode(self, text, add_special_tokens=False):
            assert not add_special_tokens
            return list(text.encode())
        def apply_chat_template(self, *args, **kwargs):
            raise AssertionError("Chat formatting must never be used")

    rows = [make_row(dict(question=f"What is {i} + 1?", answer=f"#### {i+1}"), i) for i in range(4)]
    path = tmp_path / "data.parquet"
    datasets.Dataset.from_list(rows).to_parquet(path)
    tokenizer = Tokenizer()
    dataset = CompletionDataset(str(path), tokenizer, SimpleNamespace(max_prompt_length=512, filter_overlong_prompts=True))
    text, tokens = encode_completion(tokenizer, rows[0]["prompt"][1]["content"])
    row = dataset[0]
    assert row["raw_prompt_ids"] == tokens
    assert row["input_ids"][row["attention_mask"].bool()].tolist() == tokens
    assert row["position_ids"][row["attention_mask"].bool()].tolist() == list(range(len(tokens)))
    assert row["extra_info"]["completion_prompt"] == text
    assert "####" not in text
    restored = pickle.loads(pickle.dumps(dataset))
    torch.testing.assert_close(restored[0]["input_ids"], row["input_ids"])
    loader = StatefulDataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    iterator = iter(loader)
    next(iterator)
    state = loader.state_dict()
    expected = next(iterator)["input_ids"]
    resumed = StatefulDataLoader(restored, batch_size=1, shuffle=False, num_workers=0)
    resumed.load_state_dict(state)
    torch.testing.assert_close(next(iter(resumed))["input_ids"], expected)


def test_pinned_gae_ppo_and_critic_fit():
    pytest.importorskip("verl")
    from math_rl.ppo_checks import numerical_checks
    numerical_checks()


def test_actual_critic_uses_causal_pre_token_values():
    pytest.importorskip("verl")
    import torch
    from transformers import Qwen2Config, Qwen2ForTokenClassification
    from verl.workers.critic.dp_critic import DataParallelPPOCritic

    torch.manual_seed(42)
    config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                        num_labels=1, classifier_dropout=0., attention_dropout=0.)
    model = Qwen2ForTokenClassification(config).eval()
    critic = object.__new__(DataParallelPPOCritic)
    critic.critic_module = model
    critic.device_name = "cpu"
    critic.use_remove_padding = False
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    batch = dict(input_ids=ids, responses=ids[:, -3:], attention_mask=torch.ones_like(ids),
                 position_ids=torch.arange(6).unsqueeze(0))
    first = critic._forward_micro_batch(batch)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        expected = model(input_ids=ids, attention_mask=batch["attention_mask"],
                         position_ids=batch["position_ids"]).logits[:, 2:5, 0]
    torch.testing.assert_close(first, expected)
    changed = ids.clone()
    changed[:, 3:] = torch.tensor([8, 9, 10])
    later = critic._forward_micro_batch(dict(batch, input_ids=changed, responses=changed[:, -3:]))
    torch.testing.assert_close(first[:, 0], later[:, 0])
    assert not torch.equal(first[:, 1:], later[:, 1:])


def test_hydra_config_and_external_dataset_hook():
    pytest.importorskip("verl")
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from verl.utils.import_utils import load_extern_type
    from math_rl.completion_dataset import CompletionDataset
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        config = compose(config_name="ppo")
        OmegaConf.resolve(config)
    assert config.algorithm.adv_estimator == "gae"
    assert config.critic.model.path == config.actor_rollout_ref.model.path
    assert config.critic.checkpoint.load_contents == ["model", "optimizer", "extra"]
    assert config.actor_rollout_ref.rollout.n == 1
    assert config.critic.optim.lr == 1e-5
    assert load_extern_type(str(ROOT / config.data.custom_cls.path), config.data.custom_cls.name) is CompletionDataset


def test_critic_accumulation_weights_valid_tokens_equally():
    pytest.importorskip("verl")
    import torch
    from verl.trainer.ppo.core_algos import compute_value_loss
    values = torch.tensor([[.2, .2, .2], [.2, .2, .2]], requires_grad=True)
    mask = torch.tensor([[1., 0., 0.], [1., 1., 1.]])
    loss = sum(compute_value_loss(values[i:i+1], torch.zeros(1, 3), torch.zeros(1, 3),
                                  mask[i:i+1], .5, "seq-mean-token-sum-norm")[0]
               for i in range(2)) / 2
    loss.backward()
    torch.testing.assert_close(values.grad[mask.bool()], torch.full((4,), .2 / 6))
    assert values.grad[~mask.bool()].abs().sum() == 0


def test_reward_manager_assigns_only_terminal_rewards():
    pytest.importorskip("verl")
    import numpy as np
    import torch
    from verl import DataProto
    from verl.workers.reward_manager.batch import BatchRewardManager
    from verl.utils.torch_functional import get_response_mask
    responses = torch.tensor([[7, 2, 2], [8, 9, 10]])
    mask = get_response_mask(responses, eos_token=2, dtype=torch.long)
    assert mask.tolist() == [[1, 1, 0], [1, 1, 1]]
    data = DataProto.from_dict(tensors=dict(prompts=torch.ones(2, 2, dtype=torch.long), responses=responses,
        attention_mask=torch.cat([torch.ones(2, 2, dtype=torch.long), mask], dim=1)),
        non_tensors=dict(data_source=np.array(["gsm8k"] * 2, dtype=object),
                         reward_model=np.array([dict(ground_truth="1")] * 2, dtype=object)))
    tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: str(ids.tolist()))
    manager = BatchRewardManager(tokenizer, 0, lambda **kwargs: [dict(score=1), dict(score=0)])
    scored = manager(data, return_dict=True)["reward_tensor"]
    torch.testing.assert_close(scored, torch.tensor([[0., 1., 0.], [0., 0., 0.]]))


def test_execution_report_fails_when_critic_state_missing(tmp_path):
    config = dict(algorithm=dict(adv_estimator="gae"), data=dict(train_batch_size=2),
                  trainer=dict(total_training_steps=2, n_gpus_per_node=1, nnodes=1))
    (tmp_path / "run.json").write_text(json.dumps(dict(config=config, provenance={})))
    attempt = tmp_path / "attempts/first"
    attempt.mkdir(parents=True)
    (attempt / "exit.json").write_text(json.dumps(dict(returncode=0)))
    (attempt / "console.log").write_text("\n".join(
        f"step:{i} - actor/grad_norm:0.1 - actor/pg_loss:-0.01 - critic/grad_norm:0.2 - critic/vf_loss:0.1"
        for i in (1, 2)))
    (tmp_path / "rollouts").mkdir()
    for i in (1, 2):
        (tmp_path / f"rollouts/{i}.jsonl").write_text('\n'.join(json.dumps(dict(score=s)) for s in (0, 1)))
    checkpoint = tmp_path / "global_step_2"
    checkpoint.mkdir()
    (checkpoint / "data.pt").write_bytes(b"state")
    for role in ("actor", "critic"):
        (checkpoint / role).mkdir()
        for kind in ("model", "optim", "extra_state"):
            (checkpoint / role / f"{kind}_world_size_1_rank_0.pt").write_bytes(b"state")
    command = [sys.executable, str(ROOT / "scripts/report_ppo.py"), str(tmp_path)]
    assert subprocess.run(command, capture_output=True).returncode == 0
    (checkpoint / "critic/optim_world_size_1_rank_0.pt").unlink()
    assert subprocess.run(command, capture_output=True).returncode == 1
    report = json.loads((tmp_path / "ppo-report.json").read_text())
    assert not report["gates"]["final_checkpoint_complete"]
