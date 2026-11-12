"""Prefix-only features from the unmodified official LLM2Vec backbone.

No PPO hooks: freeze this encoder and learn a NEW compressor before online use.
"""
import torch
import math


def prefix_inputs(prompt_ids, answer_ids, starts):
    """Physical truncation BEFORE a pre-action chunk boundary; preserve IDs."""
    prompt, answer = list(prompt_ids), list(answer_ids)
    starts = list(starts)
    if (not prompt or not answer or not starts or starts[0] != 0
            or any(type(p) is not int or not 0 <= p < len(answer) for p in starts)
            or starts != sorted(set(starts))):
        raise ValueError('Require nonempty prompt/answer and ordered pre-action starts beginning at zero')
    return [prompt + answer[:p] for p in starts]


def load_encoder(path, device='cpu', dtype=torch.float32, bidirectional=True, attention='eager'):
    from llm2vec import LLM2Vec
    wrapper = LLM2Vec.from_pretrained(str(path), enable_bidirectional=bidirectional,
        torch_dtype=dtype, attn_implementation=attention)
    model = wrapper.model.to(device).eval().requires_grad_(False)
    if model.config.model_type != 'qwen2':
        raise ValueError('This initial adapter is validated for Qwen2 architecture only')
    return model, wrapper.tokenizer


@torch.inference_mode()
def block_features(model, sequences, layers, pad_id, batch_size=4):
    """Last available token at selected blocks BEFORE final norm, then per-block RMS.

    Prefixes are separate batch elements, not packed into one attention sequence.
    No dummy future token and no causal KV reuse. No re-tokenization/truncation.
    """
    sequences, layers = [list(s) for s in sequences], list(layers)
    if (not sequences or any(not s for s in sequences) or batch_size < 1
            or not layers or len(set(layers)) != len(layers)
            or any(type(l) is not int or not 0 <= l < len(model.layers) for l in layers)):
        raise ValueError('Invalid prefix batch or selected blocks')
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Encoder must be eval and frozen before feature extraction')
    if any(len(s) > model.config.max_position_embeddings for s in sequences):
        raise ValueError('Prefix exceeds model context; refusing silent truncation')
    device = next(model.parameters()).device
    result = []
    for offset in range(0, len(sequences), batch_size):
        batch = sequences[offset:offset+batch_size]
        lengths = torch.tensor([len(s) for s in batch], device=device)
        ids = torch.full((len(batch), int(lengths.max())), pad_id, device=device, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i,s in enumerate(batch):
            ids[i,:len(s)] = torch.tensor(s,device=device)
            mask[i,:len(s)] = 1
        captured, hooks = {}, []
        def capture(layer):
            def hook(module, inputs, output):
                hidden = output[0] if isinstance(output,tuple) else output
                x = hidden[torch.arange(len(batch),device=device), lengths-1].float()
                captured[layer] = (x / (x.square().mean(-1,keepdim=True)+model.config.rms_norm_eps).sqrt()).cpu()
            return hook
        try:
            for layer in layers:
                hooks.append(model.layers[layer].register_forward_hook(capture(layer)))
            model(input_ids=ids,attention_mask=mask,
                  position_ids=(mask.cumsum(-1)-1).clamp_min(0),use_cache=False,return_dict=True)
        finally:
            for h in hooks:h.remove()
        if set(captured) != set(layers):raise ValueError('Missing encoder block outputs')
        result.append(torch.cat([captured[l] for l in layers],dim=1))
    x = torch.cat(result)
    if not torch.isfinite(x).all():raise ValueError('Nonfinite encoder features')
    return x


@torch.inference_mode()
def attention_probe(model, ids):
    """Check ACTUAL first-token sensitivity to a later available token.

    is_causal=False metadata alone is insufficient to prove the mask changed.
    """
    device = next(model.parameters()).device
    a = torch.tensor([ids],device=device)
    b = a.clone(); b[0,-1] = (int(b[0,-1])+7) % model.config.vocab_size
    def hidden(x):
        return model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False,
                     return_dict=True).last_hidden_state[0,0].float()
    return float((hidden(a)-hidden(b)).abs().max())


def require_bidirectional(sensitivity, causal_sensitivity):
    if (not all(math.isfinite(v) and v>=0 for v in (sensitivity,causal_sensitivity))
            or causal_sensitivity>1e-5 or sensitivity<=max(1e-5,10*causal_sensitivity)):
        raise RuntimeError('Official backend did not demonstrate bidirectional attention relative to causal control. Do not train SLICE or launch PPO; inspect report. No upstream patch applied.')
