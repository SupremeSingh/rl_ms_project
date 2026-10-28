"""Frozen hidden-state features shared by offline critics and online PPO.

These transforms use ordinary frozen transformer activations.
"""
import torch


def rms_coordinates(hidden, eps):
    """Per-token RMS normalization without the model's learned norm gain."""
    return hidden * torch.rsqrt(hidden.square().mean(-1, keepdim=True) + eps)


def isotropic_normalization(x):
    """Training-only centering and a single scalar scale across coordinates."""
    mean = x.mean(0)
    scale = ((x-mean).double().square().mean().sqrt().clamp_min(1e-8)).float()
    return mean, scale.expand_as(mean).clone()
