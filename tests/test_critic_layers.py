from types import SimpleNamespace
import torch
from torch import nn
from math_rl.critic_layers import layer_spec, sampled_layer_states, copy_backbone_state


class Block(nn.Module):
    def forward(self,x):return (x.cumsum(1)+1,)


class Backbone(nn.Module):
    def __init__(self):
        super().__init__();self.emb=nn.Embedding(32,3);self.layers=nn.ModuleList([Block() for _ in range(6)]);self.norm=nn.LayerNorm(3)
    def forward(self,input_ids,**kwargs):
        x=self.emb(input_ids)
        for layer in self.layers:x=layer(x)[0]
        return SimpleNamespace(last_hidden_state=self.norm(x))


def test_layers_are_causal_and_final_matches_original():
    model=Backbone().eval();tokens=torch.tensor([[1,2,3,4,5,6]])
    positions=[0,1,2];a=sampled_layer_states(model,tokens,3,3,positions)
    changed=tokens.clone();changed[0,-1]=17
    b=sampled_layer_states(model,changed,3,3,positions)
    assert layer_spec(28)=={'third':10,'two_thirds':19,'final':28}
    for name in a:
        assert a[name].shape==(3,3)
        torch.testing.assert_close(a[name],b[name])
    torch.testing.assert_close(a['final'],model(tokens).last_hidden_state[0,[2,3,4]])
    assert all(not layer._forward_hooks for layer in model.layers)
    assert not model.norm._forward_hooks


def test_refresh_copies_actor_backbone_only():
    actor=Backbone();encoder=Backbone()
    state={'model.'+k:v for k,v in actor.state_dict().items()};state['lm_head.weight']=torch.zeros(2,3)
    copy_backbone_state(encoder,state)
    for k,v in actor.state_dict().items():torch.testing.assert_close(encoder.state_dict()[k],v)
    assert not encoder.training and all(not p.requires_grad for p in encoder.parameters())
