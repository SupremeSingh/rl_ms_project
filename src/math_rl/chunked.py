"""Chunk-state credit assignment. Same segmentation/broadcast as VinePPO, learned values.

No auxiliary rollouts: this is NOT the VinePPO Monte Carlo estimator. gamma=1,
GAE runs over reasoning chunks, then expands to token actions before PPO whitening.
"""
import torch
from slice_critic.chunks import token_chunks


def plans_for_batch(tokenizer, responses, mask, mode):
    if responses.shape != mask.shape:
        raise ValueError('Response/mask shape mismatch')
    lengths = mask.sum(-1).long()
    expected = torch.arange(mask.shape[1], device=mask.device)[None] < lengths[:, None]
    if (lengths < 1).any() or not torch.equal(mask.bool(), expected):
        raise ValueError('Chunking requires nonempty right-padded responses')
    return [token_chunks(tokenizer, row[:int(n)].detach().cpu().tolist(), mode)
            for row, n in zip(responses, lengths, strict=True)]


def compress_rewards(rewards, plans):
    width = max(len(p.starts) for p in plans)
    result = rewards.new_zeros((len(plans), width))
    mask = rewards.new_zeros(result.shape)
    for i, plan in enumerate(plans):
        if plan.tokens > rewards.shape[1]:
            raise ValueError('Chunk tokens exceed reward width')
        for j, (a, b) in enumerate(zip(plan.starts, plan.ends)):
            result[i, j] = rewards[i, a:b].sum()
        mask[i, :len(plan.starts)] = 1
    return result, mask


@torch.no_grad()
def chunk_credit(values, rewards, mask, plans, gae_lambda=.95):
    """values are token-shaped; only causal pre-chunk values are read.

    Returns chunk GAE advantages, lambda-return targets, boundary mask,
    and piecewise-constant values, all token-shaped. Whiten separately at token level.
    """
    if isinstance(gae_lambda, bool) or not 0 <= gae_lambda <= 1:
        raise ValueError('GAE lambda must be in [0,1]')
    if values.shape != rewards.shape or mask.shape != rewards.shape or len(plans) != len(mask):
        raise ValueError('Chunk credit batch shapes disagree')
    advantages = torch.zeros_like(values)
    returns = torch.zeros_like(values)
    expanded = torch.zeros_like(values)
    boundaries = torch.zeros_like(mask)
    for i, plan in enumerate(plans):
        if int(mask[i].sum()) != plan.tokens:
            raise ValueError('Plan differs from valid response tokens')
        v = values[i, list(plan.starts)]
        r = torch.stack([rewards[i, a:b].sum() for a, b in zip(plan.starts, plan.ends)])
        next_v = torch.cat([v[1:], v.new_zeros(1)])
        delta = r + next_v - v  # gamma=1, terminal value zero including capped answers
        chunk_advantages = torch.zeros_like(v)
        carry = v.new_zeros(())
        for j in range(len(v)-1, -1, -1):
            carry = delta[j] + gae_lambda * carry
            chunk_advantages[j] = carry
        advantages[i, :plan.tokens] = plan.expand(chunk_advantages)
        returns[i, :plan.tokens] = plan.expand(chunk_advantages + v)
        expanded[i, :plan.tokens] = plan.expand(v)
        boundaries[i, list(plan.starts)] = 1
    if not all(torch.isfinite(x).all() for x in (advantages, returns, expanded)):
        raise ValueError('Nonfinite chunk credit')
    return advantages, returns, boundaries, expanded
