"""Explicit readout transforms; default leaves historical feature coordinates intact."""
import numpy as np
import torch
from slice_critic.features import rms_coordinates


def normalize_readout(features, mode='none', eps=1e-6):
    if mode == 'none':
        return features
    if mode != 'rms' or not np.isfinite(eps) or eps <= 0:
        raise ValueError('Unknown readout normalization or invalid RMS epsilon')
    x = torch.as_tensor(np.array(features, copy=True)).float()
    if x.ndim != 2 or not torch.isfinite(x).all():
        raise ValueError('Expected finite readout features')
    return rms_coordinates(x, eps).numpy().copy()
