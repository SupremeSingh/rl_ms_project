"""Compatibility adapter to the standalone slice-critic package."""
from slice_critic import encode_rollout_features as bottleneck_prefixes, load_ppo_features as load_filter

__all__ = ['bottleneck_prefixes', 'load_filter']
