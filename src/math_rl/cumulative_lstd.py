"""Compatibility adapter to the standalone slice-critic package."""
from slice_critic.cumulative import (
    CumulativeLSTD, encoder_epoch, raw_design, trajectory_statistics
)

__all__ = ['CumulativeLSTD', 'encoder_epoch', 'raw_design', 'trajectory_statistics']
