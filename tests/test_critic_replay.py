import numpy as np
import pytest
import torch

from math_rl.critic_replay import CriticReplay
from math_rl.online_critic import cross_fitted_values, frozen_prefixes, question_folds


def batch():
    g = torch.Generator().manual_seed(6)
    states = [torch.randn(3, 4, generator=g) for _ in range(8)]
    rewards = torch.zeros(8, 4)
    rewards[:, 2] = torch.arange(8) % 2
    mask = torch.tensor([[1, 1, 1, 0]] * 8)
    return states, rewards, mask, [f'q{i}' for i in range(8)]


def test_fifo_whole_answers_age_and_owned_storage():
    b = CriticReplay(3, 7, 2)
    s = torch.ones(2, 4, requires_grad=True)
    b.append([s, s], [0, 1], ['a', 'b'], 1)
    with torch.no_grad():
        s.zero_()
    assert b.entries[0].states.sum() == 8
    assert not b.entries[0].states.requires_grad
    b.append([s], [0], ['c'], 2)
    b.append([torch.ones(3, 4)], [1], ['d'], 3)
    assert [e.question for e in b.entries] == ['b', 'c', 'd']
    assert b.metrics()['transitions'] == 7
    b.append([s], [1], ['e'], 6)
    assert [e.question for e in b.entries] == ['e']
    assert b.metrics()['evicted_answers'] == 4
    with pytest.raises(ValueError, match='increasing'):
        b.append([s], [1], ['f'], 6)
    with pytest.raises(ValueError, match='entire fresh batch'):
        b.append([torch.ones(8, 4)], [1], ['g'], 7)
    assert b.last_step == 6


@pytest.mark.parametrize('method', ['ridge', 'lstd'])
def test_first_step_matches_fresh_and_history_is_used_without_label_leakage(method):
    states, rewards, mask, ids = batch()
    base, _, _ = cross_fitted_values(states, rewards, mask, ids, method)
    b = CriticReplay(24, 100, 4)
    first, _, _ = cross_fitted_values(states, rewards, mask, ids, method, replay=b, step=1)
    torch.testing.assert_close(first, base)
    # Different seed changes fold membership: old labels of held-out questions
    # must STILL be excluded, not merely their previous fold assignment.
    folds = question_folds(ids, 19)
    other = CriticReplay(24, 100, 4)
    changed = rewards.sum(-1).tolist()
    for i in np.flatnonzero(folds == 0):
        changed[i] = 1 - changed[i]
    other.append(states, changed, ids, 1)
    new_rewards = rewards.clone()
    new_rewards[:, 2] = 1 - new_rewards[:, 2]
    a, heads, metrics = cross_fitted_values(states, new_rewards, mask, ids, method,
                                          seed=19, replay=b, step=2)
    c, _, _ = cross_fitted_values(states, new_rewards, mask, ids, method,
                                 seed=19, replay=other, step=2)
    torch.testing.assert_close(a[folds == 0], c[folds == 0])
    assert metrics['replay']['historical_answers'] == 8
    assert all(f['historical_answers'] == 4 for f in metrics['folds'])
    assert set(heads[0]['train_questions']).isdisjoint({ids[i] for i in np.flatnonzero(folds == 0)})
    fresh, _, _ = cross_fitted_values(states, new_rewards, mask, ids, method, seed=19)
    assert not torch.allclose(a, fresh)
    assert a.shape == rewards.shape and not a.requires_grad
    assert not torch.count_nonzero(a[:, 3])


def test_frozen_encoder_matches_pre_action_states_with_padding_and_future_changes():
    from transformers import Qwen2Config, Qwen2Model
    config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)
    config._attn_implementation = 'eager'
    encoder = Qwen2Model(config).eval().requires_grad_(False)
    ids = torch.tensor([[0, 1, 2, 3, 4, 5, 0]])
    mask = torch.tensor([[0, 1, 1, 1, 1, 1, 0]])
    result = frozen_prefixes(encoder, ids, mask, 3)[0]
    for i in range(2):
        with torch.no_grad():
            expected = encoder(ids[:, 1:4+i]).last_hidden_state[0, -1]
        torch.testing.assert_close(torch.from_numpy(result[i]), expected)
    ids[0, 4:] = 9
    changed = frozen_prefixes(encoder, ids, mask, 3)[0]
    np.testing.assert_allclose(result[0], changed[0], atol=1e-6)


def test_run_arguments_limit_replay_to_linear_critics(tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    from math_comparison import command
    for method in ['lstd', 'ridge', 'ppo', 'grpo']:
        cmd = command(tmp_path, method, 42, 3, feature_mode='frozen', replay_capacity=256)
        linear = method in ('lstd', 'ridge')
        assert f'experiment.replay_capacity={256 if linear else 0}' in cmd
        assert f'actor_rollout_ref.model.critic_feature_mode={"frozen" if linear else "actor"}' in cmd


def test_verl_hook_replay_keeps_actor_batch_fresh(tmp_path, monkeypatch):
    pytest.importorskip('verl')
    from types import SimpleNamespace as NS
    from verl import DataProto
    from verl.trainer.ppo import ray_trainer
    from math_rl.math_trainer import MathTrainer
    import json

    states, rewards, mask, ids = batch()
    trainer = object.__new__(MathTrainer)
    trainer.method, trainer.linear, trainer.output = 'lstd', True, tmp_path
    trainer.config = NS(experiment=NS(critic_alpha=.01, critic_lambda=.99,
        replay_capacity=24, replay_max_transitions=100, replay_max_age=4),
        actor_rollout_ref=NS(model=NS(critic_feature_mode='frozen')), data=NS(seed=42))
    trainer.total_training_steps = 2

    def fit(self):
        for step in (1, 2):
            self.global_steps = step
            packed = np.empty(8, dtype=object)
            for i, x in enumerate(states):
                packed[i] = x.numpy()
            data = DataProto.from_dict(tensors=dict(token_level_rewards=rewards.clone(), response_mask=mask.clone()),
                non_tensors=dict(pretoken_features=packed,
                    extra_info=np.array([dict(prompt_id=q) for q in ids], dtype=object)))
            result = ray_trainer.compute_advantage(data, 'gae', gamma=1., lam=.95, config={})
            assert len(result.batch['advantages']) == 8
            assert torch.equal(result.batch['token_level_rewards'], rewards)
            assert 'pretoken_features' not in result.non_tensor_batch
        self.global_steps = 3
    monkeypatch.setattr(ray_trainer.RayPPOTrainer, 'fit', fit)
    trainer.fit()
    rows = [json.loads(s) for s in (tmp_path / 'linear-critic/metrics.jsonl').read_text().splitlines()]
    assert [r['replay']['historical_answers'] for r in rows] == [0, 8]
