import copy
from pathlib import Path
import sys
import pytest
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/encoder'))
from unpadded_adapter import make_unpadded,require_singleton


class Tokenizer:
    def __call__(self,text,**kw):return {'input_ids':[int(s) for s in text.split()]}


class Base(nn.Module):
    def __init__(self):
        super().__init__();self.embedding=nn.Embedding(64,4);self.tokenizer=Tokenizer();self.max_length=10;self.skip_instruction=False;self.seen=[]
    def tokenize(self,texts):
        assert len(texts)==1
        ids=torch.tensor([self.tokenizer(texts[0])['input_ids']])
        return dict(input_ids=ids,attention_mask=torch.ones_like(ids))
    def forward(self,features):
        self.seen.append(features['input_ids'].shape)
        return self.embedding(features['input_ids']).mean(1)


def test_unpadded_outputs_and_joint_gradients_match_official_singletons():
    torch.manual_seed(54);model=make_unpadded(Base)();reference=Base();reference.load_state_dict(model.state_dict())
    texts=['1 2','3 4 5 6','7']
    features=model.tokenize(texts);actual=model(features)
    expected=torch.cat([reference(reference.tokenize([text])) for text in texts])
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    actual.square().sum().backward();expected.square().sum().backward()
    assert model.embedding.weight.grad.norm()>0
    torch.testing.assert_close(model.embedding.weight.grad,reference.embedding.weight.grad)
    assert model.seen==[torch.Size([1,2]),torch.Size([1,4]),torch.Size([1,1])]


def test_padding_and_silent_truncation_are_rejected():
    model=make_unpadded(Base)()
    with pytest.raises(ValueError,match='truncation'):model.tokenize([' '.join(['1']*11)])
    with pytest.raises(ValueError,match='Padding'):require_singleton(dict(input_ids=torch.tensor([[1,0]]),attention_mask=torch.tensor([[1,0]])))
    with pytest.raises(ValueError,match='one nonempty'):require_singleton(dict(input_ids=torch.ones(2,3),attention_mask=torch.ones(2,3)))
