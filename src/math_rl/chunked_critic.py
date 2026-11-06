"""Restrict the conventional critic's loss to causal chunk-start states.

Do not change attention masks: model attention still sees every real token.
The pinned critic selects a fixed set of batch keys, so derive the loss mask in
its microbatch forward rather than relying on an extra key surviving selection.
"""
from contextlib import contextmanager
from math_rl.chunked import plans_for_batch


@contextmanager
def boundary_loss(critic, tokenizer, mode, core_algos):
    original_forward = critic._forward_micro_batch
    original_loss = core_algos.compute_value_loss
    had_override = '_forward_micro_batch' in critic.__dict__
    previous_override = critic.__dict__.get('_forward_micro_batch')
    current = None

    def forward(batch):
        nonlocal current
        mask = batch['attention_mask'][:, -batch['responses'].shape[1]:]
        plans = plans_for_batch(tokenizer, batch['responses'], mask, mode)
        current = mask.new_zeros(mask.shape)
        for i, p in enumerate(plans):
            current[i, list(p.starts)] = 1
        return original_forward(batch)

    def loss(*args, **kwargs):
        if args or current is None or current.shape != kwargs['response_mask'].shape:
            raise RuntimeError('Pinned critic loss/forward contract changed')
        kwargs['response_mask'] = current.to(kwargs['response_mask'])
        # One training target per reasoning state, not one per token in the chunk.
        kwargs['loss_agg_mode'] = 'seq-mean-token-mean'
        return original_loss(**kwargs)

    critic._forward_micro_batch = forward
    core_algos.compute_value_loss = loss
    try:
        yield
    finally:
        core_algos.compute_value_loss = original_loss
        if had_override:
            critic._forward_micro_batch = previous_override
        else:
            del critic._forward_micro_batch
