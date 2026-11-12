from types import SimpleNamespace
import pytest
import torch
from torch import nn
from math_rl.encoder_features import prefix_inputs, block_features, attention_probe, require_bidirectional


class Block(nn.Module):
    def forward(self,x,mask):
        context=(x*mask[:,:,None]).sum(1,keepdim=True)/mask.sum(1)[:,None,None]
        return (x+context,)


class ToyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=nn.Embedding(40,8)
        self.layers=nn.ModuleList([Block(),Block()])
        self.config=SimpleNamespace(max_position_embeddings=100,rms_norm_eps=1e-6,vocab_size=40)
        self.seen=[]
    def forward(self,input_ids,attention_mask,**kwargs):
        self.seen.append(input_ids.detach().clone())
        x=self.embedding(input_ids)
        for block in self.layers:x=block(x,attention_mask)[0]
        return SimpleNamespace(last_hidden_state=x)


def model():return ToyEncoder().eval().requires_grad_(False)


def test_prefix_boundaries_exclude_current_chunk_and_preserve_ids():
    assert prefix_inputs([1,2],[3,4,5,6],[0,2])==[[1,2],[1,2,3,4]]
    for starts in ([1],[0,4],[0,2,2],[0,-1]):
        with pytest.raises(ValueError):prefix_inputs([1,2],[3,4,5,6],starts)


def test_unseen_suffix_and_padding_do_not_change_features():
    encoder=model()
    a=prefix_inputs([1,2],[3,4,5,6],[0,2])
    b=prefix_inputs([1,2],[3,4,30,31,32],[0,2])
    x=block_features(encoder,a,[1,0],pad_id=0)
    y=block_features(encoder,b,[1,0],pad_id=0)
    torch.testing.assert_close(x,y,rtol=0,atol=0)
    torch.testing.assert_close(x,block_features(encoder,a,[1,0],pad_id=39,batch_size=1))
    assert x.shape==(2,16)
    assert all(5 not in ids and 6 not in ids for ids in encoder.seen)


def test_probe_detects_real_bidirectional_dependency():
    assert attention_probe(model(),[1,2,3,4])>1e-5


def test_metadata_without_actual_bidirectionality_cannot_pass_gate():
    for bidir,causal in [(0.,0.),(.001,.001),(float('nan'),0.)]:
        with pytest.raises(RuntimeError,match='bidirectional'):require_bidirectional(bidir,causal)
    require_bidirectional(.001,0.)


def test_reject_training_encoder_and_silent_context_truncation():
    encoder=model()
    encoder.train()
    with pytest.raises(ValueError,match='frozen'):block_features(encoder,[[1,2]],[0],0)
    encoder.eval()
    with pytest.raises(ValueError,match='context'):block_features(encoder,[[1]*101],[0],0)
    with pytest.raises(ValueError):block_features(encoder,[[1]],[2],0)


def test_capture_hooks_removed_after_failure():
    encoder=model()
    def fail(*a,**kw):raise RuntimeError('failure')
    encoder.forward=fail
    with pytest.raises(RuntimeError):block_features(encoder,[[1,2]],[0,1],0)
    assert all(not block._forward_hooks for block in encoder.layers)


def test_default_execution_never_pads_or_batches_prefixes():
    encoder=model()
    block_features(encoder,[[1,2],[1,2,3,4]],[0],pad_id=39)
    assert [list(t.shape) for t in encoder.seen]==[[1,2],[1,4]]
    assert all(39 not in t for t in encoder.seen)
