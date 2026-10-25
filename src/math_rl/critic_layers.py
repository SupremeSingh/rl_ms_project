"""Compatibility adapter to the standalone slice-critic package."""
from slice_critic.layers import (
    layer_spec, sampled_layer_states, copy_backbone_state, sampled_block_states
)

__all__ = ['layer_spec', 'sampled_layer_states', 'copy_backbone_state', 'sampled_block_states']
