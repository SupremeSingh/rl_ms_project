"""GRPO shares the audited scoring path but needs group-relative advantages."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from math_rl.train import validate_pilot

ROOT = Path(__file__).resolve().parents[1]


def test_grpo_rejects_single_sample_or_wrong_algorithm():
    config = dict(algorithm=dict(adv_estimator="grpo"), actor_rollout_ref=dict(
        rollout=dict(n=4, temperature=1., top_p=1., top_k=-1)))
    validate_pilot(config, "grpo")
    config["actor_rollout_ref"]["rollout"]["n"] = 1
    with pytest.raises(ValueError, match="at least two"):
        validate_pilot(config, "grpo")
    config["algorithm"]["adv_estimator"] = "gae"
    with pytest.raises(ValueError, match="grpo advantages"):
        validate_pilot(config, "grpo")


def test_pinned_grpo_normalization():
    pytest.importorskip("verl")
    from math_rl.ppo_checks import grpo_numerical_checks
    grpo_numerical_checks()


def test_grpo_inherits_reward_and_prompt_and_disables_critic(monkeypatch):
    pytest.importorskip("verl")
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer, Role
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        ppo = compose(config_name="ppo")
        grpo = compose(config_name="grpo")
        OmegaConf.resolve(ppo)
        OmegaConf.resolve(grpo)
    validate_pilot(OmegaConf.to_container(grpo), "grpo")
    assert ppo.custom_reward_function == grpo.custom_reward_function
    assert grpo.custom_reward_function.path == "src/math_rl/ppo_reward.py"
    assert grpo.reward_model.reward_manager == "batch"
    assert grpo.data.custom_cls == ppo.data.custom_cls
    assert grpo.actor_rollout_ref.model == ppo.actor_rollout_ref.model
    assert grpo.actor_rollout_ref.rollout.n == 4
    assert grpo.data.train_batch_size == 8
    assert grpo.actor_rollout_ref.actor.ppo_mini_batch_size == 8
    # Execute the pinned trainer's real algorithm selection/config validation.
    monkeypatch.setattr(RayPPOTrainer, "_create_dataloader", lambda *args: None)
    trainer = RayPPOTrainer(config=grpo, tokenizer=None,
        role_worker_mapping={Role.ActorRollout: object, Role.Critic: object},
        resource_pool_manager=None, ray_worker_group_cls=None)
    assert not trainer.use_critic
    assert not trainer.use_reference_policy


def test_math_verify_rewards_reach_grpo_advantages(monkeypatch):
    pytest.importorskip("verl")
    verifier = os.environ.get("MATH_RL_TEST_VERIFIER")
    if not verifier:
        pytest.skip("Set MATH_RL_TEST_VERIFIER to an isolated Math-Verify Python")
    import numpy as np
    import torch
    from hydra import compose, initialize_config_dir
    from types import SimpleNamespace
    from verl import DataProto
    from verl.trainer.ppo.reward import load_reward_manager
    from verl.trainer.ppo.ray_trainer import compute_advantage
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        config = compose(config_name="grpo")
    response_ids = [10, 11, 10, 11, 10, 10, 10, 10]
    tokenizer = SimpleNamespace(decode=lambda ids, **kwargs:
        r"The answer is \boxed{4}." if ids[0] == 10 else "The answer is 5.")
    manager = load_reward_manager(config, tokenizer, num_examine=0)
    # VERL loads the configured hook as custom_module; use the isolated test Python.
    monkeypatch.setattr(sys.modules["custom_module"], "verifier_command",
                        lambda: [verifier, "-m", "math_rl.verifier_batch"])
    data = DataProto.from_dict(tensors=dict(
        prompts=torch.ones(8, 2, dtype=torch.long),
        responses=torch.tensor([[token, 2, 2] for token in response_ids]),
        attention_mask=torch.tensor([[1, 1, 1, 1, 0]] * 8)), non_tensors=dict(
            data_source=np.array(["gsm8k"] * 8, dtype=object),
            uid=np.array(["a"] * 4 + ["b"] * 4, dtype=object),
            reward_model=np.array([dict(ground_truth="4")] * 8, dtype=object)))
    result = manager(data, return_dict=True)
    rewards = result["reward_tensor"]
    assert rewards.sum(-1).tolist() == [1, 0, 1, 0, 1, 1, 1, 1]
    assert rewards[:, [0, 2]].sum() == 0
    data.batch["token_level_rewards"] = rewards
    data = compute_advantage(data, "grpo", config=config.algorithm)
    advantage = data.batch["advantages"]
    assert (advantage[[0, 2], :2] > 0).all()
    assert (advantage[[1, 3], :2] < 0).all()
    assert advantage[4:].abs().sum() == 0
    assert advantage[:, 2].abs().sum() == 0
    # Actual policy-loss gradients must favor accepted answers, not merely log scores.
    from verl.trainer.ppo.core_algos import compute_policy_loss
    log_probs = torch.zeros_like(advantage, requires_grad=True)
    loss = compute_policy_loss(torch.zeros_like(log_probs), log_probs, advantage,
        data.batch["response_mask"], cliprange=.2, clip_ratio_c=1e9,
        loss_agg_mode="seq-mean-token-mean")[0]
    loss.backward()
    assert (log_probs.grad[[0, 2], :2] < 0).all()
    assert (log_probs.grad[[1, 3], :2] > 0).all()
    assert log_probs.grad[4:].abs().sum() == 0
    assert log_probs.grad[:, 2].abs().sum() == 0


@pytest.mark.parametrize("mixed", [True, False])
def test_grpo_report_requires_mixed_rewards_within_a_group(tmp_path, mixed):
    config = dict(algorithm=dict(adv_estimator="grpo"), data=dict(train_batch_size=2),
        actor_rollout_ref=dict(rollout=dict(n=4)),
        trainer=dict(total_training_steps=1, n_gpus_per_node=1, nnodes=1))
    (tmp_path / "run.json").write_text(json.dumps(dict(config=config, provenance={})))
    attempt = tmp_path / "attempts/first"
    attempt.mkdir(parents=True)
    (attempt / "exit.json").write_text('{"returncode": 0}')
    (attempt / "console.log").write_text("step:1 - actor/grad_norm:0.1 - actor/pg_loss:-0.01")
    (tmp_path / "rollouts").mkdir()
    outcomes = [0, 1, 0, 1] * 2 if mixed else [0] * 4 + [1] * 4
    rows = [dict(input=f"question {i // 4}", score=score) for i, score in enumerate(outcomes)]
    (tmp_path / "rollouts/1.jsonl").write_text('\n'.join(map(json.dumps, rows)))
    checkpoint = tmp_path / "global_step_1"
    (checkpoint / "actor").mkdir(parents=True)
    (checkpoint / "data.pt").write_bytes(b"state")
    for kind in ("model", "optim", "extra_state"):
        (checkpoint / "actor" / f"{kind}_world_size_1_rank_0.pt").write_bytes(b"state")
    result = subprocess.run([sys.executable, str(ROOT / "scripts/report_ppo.py"), str(tmp_path)],
                            capture_output=True, text=True)
    assert result.returncode == (0 if mixed else 1), result.stderr
    report = json.loads(result.stdout)
    assert report["algorithm"] == "grpo"
    assert report["responses"] == 8
    assert report["gates"]["final_checkpoint_complete"]
    assert "critic_updated" not in report["gates"]
    assert report["gates"]["mixed_group_actor_update"] == mixed
