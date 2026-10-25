"""Compatibility adapter to the standalone slice-critic package."""
from slice_critic.features import (
    FEATURE_PACKAGES, rms_coordinates, isotropic_normalization, hidden_features, validate_feature_runtime, load_frozen_causal_model
)

__all__ = ['FEATURE_PACKAGES', 'rms_coordinates', 'isotropic_normalization', 'hidden_features', 'validate_feature_runtime', 'load_frozen_causal_model']
