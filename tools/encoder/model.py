"""Qwen-only LLM2Vec-style adapters; no changes to the actor or SLICE library."""
import torch
from torch import nn
from torch.nn import functional as F


class LowRank(nn.Module):
    def __init__(self, base, rank=8):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.a, a=5**.5)
        self.scale = 2.

    def forward(self, x):
        return self.base(x) + (F.linear(F.linear(x.float(), self.a), self.b)*self.scale).to(x.dtype)


def install_adapters(model, rank=8):
    model.requires_grad_(False)
    for layer in model.model.layers:
        for name in ('q_proj','k_proj','v_proj','o_proj'):
            setattr(layer.self_attn, name, LowRank(getattr(layer.self_attn,name), rank))
        for name in ('gate_proj','up_proj','down_proj'):
            setattr(layer.mlp, name, LowRank(getattr(layer.mlp,name), rank))


def adapter_state(model):
    return {k:v.detach().cpu() for k,v in model.named_parameters() if k.endswith(('.a','.b'))}


def restore_adapter(model, path):
    record = torch.load(path, map_location='cpu', weights_only=True)
    install_adapters(model,record['rank'])
    params = dict(model.named_parameters())
    if set(record['weights']) != set(adapter_state(model)):
        raise ValueError('Adapter parameter contract changed')
    for key, value in record['weights'].items():
        if value.shape != params[key].shape: raise ValueError('Adapter shape changed')
        params[key].data.copy_(value)
    return record


def forward(model, ids, bidirectional=False):
    """Inputs are UNPADDED standalone prefixes. Never pass a complete answer here.

    Qwen eager accepts an explicit 4-D additive mask; zero permits every supplied
    prefix token to attend every other one. Causal mode uses its native mask.
    """
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1:
        raise ValueError('Require one nonempty unpadded prefix')
    mask = torch.zeros((1,1,ids.shape[1],ids.shape[1]), device=ids.device,
                       dtype=next(model.parameters()).dtype) if bidirectional else torch.ones_like(ids)
    return model.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True).last_hidden_state


def masked_next_loss(model, ids, mask_token, generator, probability=.2):
    """Predict original masked token t from hidden position t-1, not from t.

    All selected inputs are replaced (no unchanged-target shortcut). Existing EOS
    serves as a mask sentinel; no vocabulary resize or actor-token changes.
    """
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 2:
        raise ValueError('MNTP requires at least two tokens')
    selected = torch.rand(ids.shape, generator=generator).to(ids.device) < probability
    selected[:,0] = False
    if not selected.any(): selected[0,-1] = True
    corrupted = ids.clone(); corrupted[selected] = mask_token
    hidden = forward(model,corrupted,True)
    positions = selected[0].nonzero().flatten()
    logits = model.lm_head(hidden[:,positions-1]).float().reshape(-1,model.config.vocab_size)
    return F.cross_entropy(logits,ids[0,positions])


def embedding(model, ids):
    return F.normalize(forward(model,ids,True).float().mean(1),dim=-1)


def contrastive_loss(left, right, temperature=.05):
    if left.shape != right.shape or len(left)<2: raise ValueError('Require at least two question-distinct positive pairs')
    logits = left@right.T/temperature
    labels = torch.arange(len(left),device=left.device)
    return (F.cross_entropy(logits,labels)+F.cross_entropy(logits.T,labels))/2


@torch.no_grad()
def attention_check(model, device):
    """Probe changed later input: native causal earlier states must stay fixed."""
    was_training = model.training; model.eval()
    a = torch.tensor([[10,11,12,13]],device=device); b = a.clone(); b[0,-1] = 14
    causal = (forward(model,a,False)[:,0]-forward(model,b,False)[:,0]).float().abs().max().item()
    bi = (forward(model,a,True)[:,0]-forward(model,b,True)[:,0]).float().abs().max().item()
    model.train(was_training)
    if causal>1e-5 or bi<=1e-6: raise ValueError(f'Attention contract failed: causal={causal}, bidirectional={bi}')
    return dict(causal_earlier_token_change=causal,bidirectional_earlier_token_change=bi)


@torch.no_grad()
def prefix_features(model, tokens, layers, raw_layer, bidirectional, device):
    """Capture last AVAILABLE token at selected blocks; no dummy future token."""
    from slice_critic.features import rms_coordinates
    states = {}; hooks=[]
    def capture(index):
        def hook(module, args, output):
            value = output[0] if isinstance(output,tuple) else output
            states[index] = value[0,-1].float().cpu().clone()
        return hook
    try:
        for index in set(layers+[raw_layer]):
            hooks.append(model.model.layers[index].register_forward_hook(capture(index)))
        forward(model,torch.tensor([tokens],device=device),bidirectional)
    finally:
        for hook in hooks: hook.remove()
    concat = torch.cat([rms_coordinates(states[i],model.config.rms_norm_eps) for i in layers])
    raw = states[raw_layer]
    return dict(concat=concat, raw=raw, rms=rms_coordinates(raw,model.config.rms_norm_eps))


@torch.no_grad()
def causal_features(model, prompt, answer, positions, layers, raw_layer, device):
    """One causal pass supplies every selected pre-action state without lookahead."""
    from slice_critic.layers import sampled_block_states
    from slice_critic.features import rms_coordinates
    ids = torch.tensor([prompt+answer], device=device)
    states = sampled_block_states(model.model, ids, len(prompt), len(answer), positions,
                                  list(dict.fromkeys(layers+[raw_layer])))
    raw = states[raw_layer].float()
    return dict(concat=torch.cat([rms_coordinates(states[i].float(),model.config.rms_norm_eps) for i in layers],1),
                raw=raw, rms=rms_coordinates(raw,model.config.rms_norm_eps))
