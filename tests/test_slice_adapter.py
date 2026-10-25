"""Existing PPO imports resolve to the standalone package implementation."""
from slice_critic import CumulativeCritic


def test_existing_adapters_share_implementations():
    from math_rl.cumulative_lstd import CumulativeLSTD
    from math_rl.sparse_bottleneck import Bottleneck
    from slice_critic.bottleneck import Bottleneck as SharedBottleneck
    from math_rl.critic_layers import sampled_block_states
    from slice_critic.layers import sampled_block_states as shared_states
    assert CumulativeCritic is CumulativeLSTD
    assert Bottleneck is SharedBottleneck
    assert sampled_block_states is shared_states
