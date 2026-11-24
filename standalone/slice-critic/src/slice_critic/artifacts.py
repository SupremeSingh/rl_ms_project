"""CPU-only validation of the versioned artifact contract before model loading."""
import math

import torch

from .bottleneck import Bottleneck
from .config import INSTRUCTION
from .storage import read_artifact, read_json


def validate_payload(manifest, checkpoint, calibration):
    """Validate metadata and tensors without loading an encoder or changing files."""
    try:
        features = manifest['features']
        width, layers = features['width'], features['layers_zero_based']
        if type(width) is not int or width < 1:
            raise ValueError('Artifact width must be a positive integer')
        if (not isinstance(layers, list) or not layers or any(type(i) is not int or i < 0 for i in layers)
                or len(set(layers)) != len(layers)):
            raise ValueError('Artifact layers must be distinct nonnegative block indices')
        if features.get('blocks_one_based', [i+1 for i in layers]) != [i+1 for i in layers]:
            raise ValueError('Artifact layer numbering disagrees')
        eps = features['rms_eps']
        if isinstance(eps, bool) or not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0:
            raise ValueError('Artifact RMS epsilon must be finite and positive')
        if features['compression'] not in ('plain', 'sparse'):
            raise ValueError('Unsupported artifact compression')
        if manifest['prompt_instruction'] != INSTRUCTION:
            raise ValueError('Artifact prompt differs from the supported completion contract')
        if not isinstance(manifest['version'], str) or not manifest['version']:
            raise ValueError('Artifact needs a version identifier')
        model = manifest['model']
        if not isinstance(model['id'], str) or not model['id'] or not isinstance(model['files'], dict) or not model['files']:
            raise ValueError('Artifact needs an encoder identity')
        for key in ('representation_training_families', 'selection_families'):
            if not isinstance(manifest[key], list) or any(not isinstance(f, str) for f in manifest[key]):
                raise ValueError('Artifact needs recorded fitting and selection families')
        if set(manifest['representation_training_families']) & set(manifest['selection_families']):
            raise ValueError('Artifact fitting and selection families overlap')
        mean, scale, weights = (checkpoint[k] for k in ('mean', 'scale', 'weights'))
        if (not all(isinstance(t, torch.Tensor) and t.is_floating_point() for t in (mean, scale, weights))
                or mean.ndim != 1 or not mean.numel() or mean.shape != scale.shape
                or weights.shape != (width+1,) or mean.numel() % len(layers)
                or features.get('input_dimension', mean.numel()) != mean.numel()):
            raise ValueError('Artifact feature or head dimensions disagree')
        if (not all(torch.isfinite(t).all() for t in (mean, scale, weights)) or (scale <= 0).any()):
            raise ValueError('Artifact normalization and head must be finite, with positive scales')
        # Inspection must not consume the caller's random stream.
        with torch.random.fork_rng(devices=[]):
            compressor = Bottleneck(mean.numel(), width)
        try:
            compressor.load_state_dict(checkpoint['compressor'], strict=True)
        except RuntimeError as exc:
            raise ValueError('Artifact compressor dimensions disagree') from exc
        if any(not torch.isfinite(p).all() for p in compressor.parameters()):
            raise ValueError('Artifact compressor must be finite')
        if calibration['method'] != 'platt' or any(
                isinstance(calibration[k], bool) or not isinstance(calibration[k], (int, float))
                or not math.isfinite(calibration[k]) for k in ('slope', 'intercept')):
            raise ValueError('Artifact needs finite Platt calibration')
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f'Malformed critic artifact: {exc}') from exc


def load_validated_artifact(path):
    manifest, checkpoint, calibration = read_artifact(path)
    validate_payload(manifest, checkpoint, calibration)
    return manifest, checkpoint, calibration


def inspect_artifact(path):
    """Check hashes and compatibility on CPU; no download, encoder or GPU required."""
    from pathlib import Path
    manifest, _, calibration = load_validated_artifact(path)
    return dict(manifest=manifest, calibration=calibration,
                evaluation=read_json(Path(path)/'evaluation.json'), validated=True)
