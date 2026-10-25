"""Frozen hidden-state features shared by offline critics and online PPO.

These transforms use ordinary transformer activations, without a Jacobian lens.
"""
import os
from importlib.metadata import version
import torch

FEATURE_PACKAGES = ('torch', 'transformers', 'numpy', 'huggingface-hub', 'tokenizers')


def rms_coordinates(hidden, eps):
    """Per-token RMS normalization without the model's learned norm gain."""
    return hidden * torch.rsqrt(hidden.square().mean(-1, keepdim=True) + eps)


def isotropic_normalization(x):
    """Training-only centering and a single scalar scale across coordinates."""
    mean = x.mean(0)
    scale = ((x-mean).double().square().mean().sqrt().clamp_min(1e-8)).float()
    return mean, scale.expand_as(mean).clone()


def hidden_features(arm, blocks, manifest, gain=None):
    """Read raw or RMS-normalized frozen blocks in the saved layer order."""
    if arm == 'hidden_two_thirds':
        return blocks[manifest['two_thirds']]
    if arm == 'final':
        if gain is None:
            raise ValueError('Final normalized features require the learned norm gain')
        return rms_coordinates(blocks[manifest['layers'][0]], manifest['rms_eps']) * gain
    if arm not in ('hidden_concat', 'rms_hidden_concat'):
        raise ValueError(f'Unknown hidden feature representation: {arm}')
    return torch.cat([
        rms_coordinates(blocks[layer], manifest['rms_eps'])
        if arm == 'rms_hidden_concat' else blocks[layer]
        for layer in manifest['layers']
    ], 1)


def validate_feature_runtime(expected):
    """Check feature-producing dependencies from a saved runtime snapshot."""
    actual = {name: version(name) for name in FEATURE_PACKAGES}
    if actual != {name: expected[name] for name in FEATURE_PACKAGES}:
        raise ValueError('Frozen feature runtime changed; use the pinned critic-training environment')
    return actual


def load_frozen_causal_model(path):
    """Offline FP32/eager extraction matching the original frozen feature recipe."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError('Run frozen feature extraction in a GPU allocation')
    if os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE') == '1':
        raise RuntimeError('Unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE for FP32 critic features')
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True,
        torch_dtype=torch.float32, attn_implementation='eager').cuda().eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    return model, tokenizer
