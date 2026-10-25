"""Detached pre-action bottleneck features for existing batched PPO integrations."""
import torch
from .inference import load_filter
from .layers import sampled_block_states
from .features import rms_coordinates


@torch.inference_mode()
def bottleneck_prefixes(encoder, ids, masks, response_width, fitted, layers, eps):
    model, mean, scale = fitted
    device = next(encoder.parameters()).device
    model.to(device)
    result = []
    try:
        for row, mask in zip(ids, masks, strict=True):
            prompt = int(mask[:-response_width].sum())
            response = int(mask[-response_width:].sum())
            tokens = row[mask.bool()].unsqueeze(0).to(device)
            blocks = sampled_block_states(encoder, tokens, prompt, response, list(range(response)), layers)
            x = torch.cat([rms_coordinates(blocks[l].float(), eps) for l in layers], 1)
            if x.shape[1] != mean.numel():
                raise ValueError('Offline/online feature dimensions differ')
            z = torch.cat([model.features(((v-mean)/scale).to(device)).cpu() for v in x.split(512)])
            if not torch.isfinite(z).all():
                raise ValueError('Nonfinite bottleneck features')
            result.append(z.numpy().copy())
    finally:
        model.cpu()
    return result
