"""Numerical checks against the pinned VERL functions used by the pilot."""
from pathlib import Path


def checkpoint_complete(path, world_size):
    path = Path(path)
    files = [path / "data.pt"]
    for role in ("actor", "critic"):
        for rank in range(world_size):
            for kind in ("model", "optim", "extra_state"):
                files.append(path / role / f"{kind}_world_size_{world_size}_rank_{rank}.pt")
    return all(p.is_file() and p.stat().st_size > 0 for p in files)


def numerical_checks():
    import torch
    from verl.trainer.ppo.core_algos import compute_gae_advantage_return, compute_policy_loss, compute_value_loss
    from verl.utils.torch_functional import masked_whiten

    # A three-token response and a two-token response with right padding.
    # V is evaluated BEFORE each action. EOS/budget exhaustion both terminate.
    rewards = torch.tensor([[0., 0., 1.], [0., 1., 0.]])
    values = torch.tensor([[.2, .3, .1], [.4, .2, 0.]], requires_grad=True)
    mask = torch.tensor([[1., 1., 1.], [1., 1., 0.]])
    advantages, returns = compute_gae_advantage_return(rewards, values, mask, .9, .8)
    expected = torch.tensor([[.58536, .738, 1.], [.756, 1., 0.]])
    torch.testing.assert_close(returns, expected)
    torch.testing.assert_close(advantages, masked_whiten(expected - values.detach(), mask))
    assert not advantages.requires_grad and not returns.requires_grad
    _, mc_returns = compute_gae_advantage_return(rewards, values, mask, 1., 1.)
    torch.testing.assert_close(mc_returns, mask)

    # At identical policies, ratio=1 and clipping/KL are zero.
    old = torch.zeros(1, 2)
    adv = torch.tensor([[1., -1.]])
    full_mask = torch.ones_like(adv)
    loss, clipped, kl, _ = compute_policy_loss(old, old, adv, full_mask,
                                               cliprange=.2, clip_ratio_c=1e9)
    torch.testing.assert_close(torch.stack([loss, clipped, kl]), torch.zeros(3))
    logp = torch.tensor([[1.5, .5]]).log().requires_grad_()
    loss, _, _, _ = compute_policy_loss(old, logp, adv, full_mask,
                                        cliprange=.2, clip_ratio_c=1e9)
    torch.testing.assert_close(loss, torch.tensor(-.2))
    loss.backward()
    torch.testing.assert_close(logp.grad, torch.zeros_like(logp))

    # Fit fixed, detached returns with the actual value loss; actor stays isolated.
    actor = torch.nn.Parameter(torch.zeros(1, 2))
    critic = torch.nn.Parameter(torch.zeros(1, 2))
    target = torch.tensor([[.2, .4]])
    optimizer = torch.optim.Adam([critic], lr=.03)
    def value_loss():
        return compute_value_loss(critic, target, torch.zeros_like(critic), full_mask,
                                  cliprange_value=.5)[0]
    initial = value_loss().item()
    for _ in range(30):
        optimizer.zero_grad()
        value_loss().backward()
        optimizer.step()
    assert value_loss().item() < initial / 10
    assert actor.grad is None and optimizer.state
    optimizer.zero_grad(set_to_none=True)
    policy_loss = compute_policy_loss(old, actor, adv, full_mask,
                                      cliprange=.2, clip_ratio_c=1e9)[0]
    policy_loss.backward()
    assert actor.grad.abs().sum() > 0 and critic.grad is None
    print("PPO numerical checks passed: GAE, masks, detached returns, clipping, critic fit, gradient isolation")
