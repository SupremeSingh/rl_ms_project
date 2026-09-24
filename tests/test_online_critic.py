import numpy as np
import pytest
import torch

from math_rl.online_critic import capture_prefixes, cross_fitted_values, fit_head, question_folds


def trajectories():
    generator = torch.Generator().manual_seed(17)
    states = [torch.randn(3 + i % 4, 3, generator=generator) for i in range(16)]
    mask = torch.zeros(16, 8)
    rewards = torch.zeros_like(mask)
    for i, state in enumerate(states):
        mask[i, :len(state)] = 1
        rewards[i, len(state) - 1] = i % 2
    ids = [f'question-{i // 2}' for i in range(16)]
    return states, rewards, mask, ids


@pytest.mark.parametrize('method', ['lstd', 'ridge'])
def test_cross_fit_never_uses_own_question_labels(method):
    states, rewards, mask, ids = trajectories()
    values, heads, metrics = cross_fitted_values(states, rewards, mask, ids, method)
    folds = question_folds(ids, 42)
    changed = rewards.clone()
    lengths = mask.sum(-1).long()
    for i in np.flatnonzero(folds == 0):
        changed[i, lengths[i] - 1] = 1 - changed[i, lengths[i] - 1]
    new, _, _ = cross_fitted_values(states, changed, mask, ids, method)
    assert torch.equal(values[folds == 0], new[folds == 0])
    assert torch.count_nonzero(values[mask == 0]) == 0
    assert not values.requires_grad
    for fold, head in enumerate(heads):
        assert set(head['train_questions']).isdisjoint({ids[i] for i in np.flatnonzero(folds == fold)})
    assert metrics['answers'] == 16 and metrics['transitions'] == int(mask.sum())
    assert all(f['relative_residual'] < 1e-8 for f in metrics['folds'])


def test_lstd_one_equals_ridge_with_matched_online_data():
    states, rewards, mask, _ = trajectories()
    a, _ = fit_head(states, rewards.sum(-1).tolist(), 'lstd', .01, 1.)
    b, _ = fit_head(states, rewards.sum(-1).tolist(), 'ridge', .01, 1.)
    torch.testing.assert_close(a['weights'], b['weights'], atol=1e-9, rtol=1e-9)


def test_fold_predictions_follow_rows_after_dp_reordering():
    states, rewards, mask, ids = trajectories()
    reference, _, _ = cross_fitted_values(states, rewards, mask, ids, 'ridge')
    order = torch.randperm(16, generator=torch.Generator().manual_seed(3))
    result, _, _ = cross_fitted_values([states[i] for i in order], rewards[order], mask[order],
                                      [ids[i] for i in order], 'ridge')
    torch.testing.assert_close(result, reference[order])


def test_rejects_nonterminal_rewards_and_bad_masks():
    states, rewards, mask, ids = trajectories()
    rewards[0, 0] = .5
    with pytest.raises(ValueError, match='terminal rewards'):
        cross_fitted_values(states, rewards, mask, ids, 'ridge')
    rewards[0, 0] = 0
    mask[0, 0] = 0
    with pytest.raises(ValueError, match='contiguous'):
        cross_fitted_values(states, rewards, mask, ids, 'ridge')


def test_real_qwen_norm_is_causal_and_pre_action():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(1)
    config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, attention_dropout=0.)
    config._attn_implementation = 'eager'
    model = Qwen2ForCausalLM(config).eval()
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
    with torch.no_grad(), capture_prefixes(model, [(3, 3)]) as captured:
        full = model(tokens, output_hidden_states=True)
    states = torch.from_numpy(captured[0])
    torch.testing.assert_close(states, full.hidden_states[-1][0, 2:5])
    for action in range(3):
        with torch.no_grad():
            prefix = model(tokens[:, :3 + action], output_hidden_states=True)
        torch.testing.assert_close(states[action], prefix.hidden_states[-1][0, -1])
    assert not model.model.norm._forward_hooks
    with pytest.raises(ValueError, match='shape/order'):
        with capture_prefixes(model, [(3, 2)]):
            model(tokens)
    assert not model.model.norm._forward_hooks


def test_pinned_gae_accepts_detached_cross_fit_values():
    pytest.importorskip('verl')
    from verl.trainer.ppo.core_algos import compute_gae_advantage_return
    states, rewards, mask, ids = trajectories()
    values, _, _ = cross_fitted_values(states, rewards, mask, ids, 'lstd')
    advantages, returns = compute_gae_advantage_return(rewards, values, mask, gamma=1., lam=.95)
    assert torch.isfinite(advantages).all() and torch.isfinite(returns).all()
    # Raw GAE returns, not whitened actor advantages, must define critic targets.
    for i, length in enumerate(mask.sum(-1).long()):
        torch.testing.assert_close(returns[i, length - 1], rewards[i, length - 1])


@pytest.mark.parametrize('method', ['ridge', 'lstd'])
def test_constant_outcomes_and_rank_deficient_features(method):
    # A small online batch can contain only failures, or only successes.
    for outcome in (0., 1.):
        states = [torch.ones(3, 5) for _ in range(8)]
        mask = torch.ones(8, 3)
        rewards = torch.zeros_like(mask)
        rewards[:, -1] = outcome
        values, _, _ = cross_fitted_values(states, rewards, mask,
            [f'q{i // 2}' for i in range(8)], method)
        torch.testing.assert_close(values, torch.full_like(values, outcome))


@pytest.mark.parametrize('method', ['ridge', 'lstd'])
def test_trainer_hook_feeds_actual_verl_gae_and_ppo_loss(tmp_path, monkeypatch, method):
    pytest.importorskip('verl')
    from types import SimpleNamespace
    from verl import DataProto
    from verl.trainer.ppo import ray_trainer, core_algos
    from math_rl.math_trainer import MathTrainer

    states, rewards, mask, ids = trajectories()
    packed = np.empty(len(states), dtype=object)
    for i, state in enumerate(states):
        packed[i] = state.numpy()
    data = DataProto.from_dict(tensors=dict(token_level_rewards=rewards, response_mask=mask),
        non_tensors=dict(pretoken_features=packed,
                        extra_info=np.array([dict(prompt_id=q) for q in ids], dtype=object)))
    trainer = object.__new__(MathTrainer)
    trainer.method, trainer.linear, trainer.output = method, True, tmp_path
    trainer.config = SimpleNamespace(experiment=SimpleNamespace(critic_alpha=.01, critic_lambda=.99),
                                     data=SimpleNamespace(seed=42))
    trainer.total_training_steps = 1
    original = ray_trainer.compute_advantage

    def fit(self):
        self.global_steps = 1
        result = ray_trainer.compute_advantage(data, 'gae', gamma=1., lam=.95, config={})
        assert 'pretoken_features' not in result.non_tensor_batch
        assert not result.batch['values'].requires_grad
        expected, returns = core_algos.compute_gae_advantage_return(
            rewards, result.batch['values'], mask, gamma=1., lam=.95)
        torch.testing.assert_close(result.batch['advantages'], expected)
        torch.testing.assert_close(result.batch['returns'], returns)
        # Exercise the pinned PPO loss, not a second implementation of PPO.
        log_prob = torch.nn.Parameter(torch.zeros_like(mask))
        optimizer = torch.optim.SGD([log_prob], lr=.01)
        loss, *_ = core_algos.compute_policy_loss(
            old_log_prob=torch.zeros_like(mask), log_prob=log_prob, advantages=expected,
            response_mask=mask, cliprange=.2, cliprange_low=.2, cliprange_high=.2,
            clip_ratio_c=1e9, loss_agg_mode='seq-mean-token-mean')
        loss.backward()
        assert torch.isfinite(log_prob.grad).all() and log_prob.grad.abs().sum() > 0
        optimizer.step()
        assert log_prob.detach().abs().sum() > 0
        self.global_steps = 2

    monkeypatch.setattr(ray_trainer.RayPPOTrainer, 'fit', fit)
    trainer.fit()
    assert trainer.completed_updates == 1
    assert (tmp_path / 'linear-critic/1.pt').is_file()
    assert ray_trainer.compute_advantage is original
    def fail(self):
        raise RuntimeError('worker failed')
    monkeypatch.setattr(ray_trainer.RayPPOTrainer, 'fit', fail)
    with pytest.raises(RuntimeError, match='worker failed'):
        trainer.fit()
    assert ray_trainer.compute_advantage is original
