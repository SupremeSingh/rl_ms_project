"""Compatibility adapter to the standalone slice-critic package."""
from slice_critic.online import (
    bottleneck_prefixes, load_filter
)

__all__ = ['bottleneck_prefixes', 'load_filter']
