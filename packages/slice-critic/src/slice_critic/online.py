"""Detached pre-action bottleneck features for existing batched PPO integrations."""
import torch
import numpy as np
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


@torch.inference_mode()
def frozen_prefixes(encoder, input_ids, attention_mask, response_width, layer="final"):
    """One causal encoder pass per unpadded answer, retaining pre-action states."""
    device = next(encoder.parameters()).device
    features = []
    for ids, mask in zip(input_ids, attention_mask, strict=True):
        prompt = int(mask[:-response_width].sum())
        response = int(mask[-response_width:].sum())
        if prompt < 1 or response < 1:
            raise ValueError('Empty frozen-feature prompt or response')
        tokens = ids[mask.bool()].unsqueeze(0).to(device)
        if layer == 'final':
            hidden = encoder(input_ids=tokens, use_cache=False, return_dict=True).last_hidden_state
        elif layer == 'two_thirds':
            from .layers import layer_spec
            captured = []
            def capture(module, inputs, output):
                captured.append(output[0] if isinstance(output, tuple) else output)
            hook = encoder.layers[layer_spec(len(encoder.layers))['two_thirds'] - 1].register_forward_hook(capture)
            try:
                encoder(input_ids=tokens, use_cache=False, return_dict=True)
            finally:
                hook.remove()
            if len(captured) != 1:
                raise ValueError('Expected exactly one intermediate feature capture')
            hidden = captured[0]
        else:
            raise ValueError('Unknown critic feature layer')
        states = hidden[0, prompt-1:prompt+response-1].float().cpu().numpy().copy()
        if len(states) != response or not np.isfinite(states).all():
            raise ValueError('Invalid frozen features')
        features.append(states)
    return features
