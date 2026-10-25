"""Compatibility adapter to the standalone slice-critic package."""
from slice_critic.bottleneck import (
    Bottleneck, loss, encode
)

__all__ = ['Bottleneck', 'loss', 'encode']
