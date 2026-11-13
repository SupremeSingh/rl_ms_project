from pathlib import Path
import sys

import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/encoder'))
from distributed_check import check_embedding_gradients


def test_gather_gate_expects_rank_sum_not_single_loss_gradient():
    reference=torch.tensor([.03,-.1,.7])
    check_embedding_gradients(reference*4,reference,4)
    for wrong in (reference,reference*2,reference*8):
        with pytest.raises(AssertionError):check_embedding_gradients(wrong,reference,4)


def test_rank_summed_gather_and_ddp_average_equal_dense_parameter_gradient():
    torch.manual_seed(812)
    world=4;batch=8;x=torch.randn(world*batch,16);y=torch.randn_like(x)
    reference=torch.nn.Linear(16,12,bias=False)
    replicas=[torch.nn.Linear(16,12,bias=False) for _ in range(world)]
    for model in replicas:model.load_state_dict(reference.state_dict())
    def loss(q,d):
        return F.cross_entropy(F.normalize(q,dim=1)@F.normalize(d,dim=1).T*20,torch.arange(len(q)))
    q=torch.cat([model(x[i*batch:(i+1)*batch]) for i,model in enumerate(replicas)])
    d=torch.cat([model(y[i*batch:(i+1)*batch]) for i,model in enumerate(replicas)])
    # The global loss is evaluated on every rank; gather backward sums them.
    (world*loss(q,d)).backward()
    loss(reference(x),reference(y)).backward()
    averaged=torch.stack([model.weight.grad for model in replicas]).mean(0)
    torch.testing.assert_close(averaged,reference.weight.grad,atol=1e-5,rtol=1e-4)
