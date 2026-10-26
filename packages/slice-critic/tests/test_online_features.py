from types import SimpleNamespace

import pytest
import torch
from torch import nn

from slice_critic import encode_hidden_rollout_features


class Block(nn.Module):
    def forward(self,x):
        return (x.cumsum(1)+.1,)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb=nn.Embedding(20,3)
        self.layers=nn.ModuleList([Block() for _ in range(6)])
        self.norm=nn.LayerNorm(3)

    def forward(self,input_ids,**kwargs):
        x=self.emb(input_ids)
        for block in self.layers:x=block(x)[0]
        return SimpleNamespace(last_hidden_state=self.norm(x))


@pytest.mark.parametrize('layer',['two_thirds','final'])
def test_raw_controls_keep_pre_action_index_norm_and_causality(layer):
    torch.manual_seed(19)
    encoder=Encoder().eval().requires_grad_(False)
    ids=torch.tensor([[1,2,3,4,5]])
    mask=torch.ones_like(ids)
    features=encode_hidden_rollout_features(encoder,ids,mask,3,layer)[0]
    x=encoder.emb(ids)
    for i,block in enumerate(encoder.layers):
        x=block(x)[0]
        if layer=='two_thirds' and i==3:break
    if layer=='final':x=encoder.norm(x)
    torch.testing.assert_close(torch.from_numpy(features),x[0,1:4])
    changed=ids.clone();changed[0,3]=9
    other=encode_hidden_rollout_features(encoder,changed,mask,3,layer)[0]
    torch.testing.assert_close(torch.from_numpy(features[:2]),torch.from_numpy(other[:2]))
