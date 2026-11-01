"""Gradient boundaries and full synthetic offline neural-critic workflow."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from slice_critic.bottleneck import Bottleneck
from slice_critic.storage import sha256

TOOLS=Path(__file__).resolve().parents[1]/'tools/diagnostics'
sys.path.insert(0,str(TOOLS))
try:
    spec=importlib.util.spec_from_file_location('slice_mlp',TOOLS/'slice_mlp.py')
    study=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study)
finally:
    sys.path.remove(str(TOOLS))


@pytest.mark.parametrize('joint',[False,True])
def test_gradient_boundary_and_parameter_updates(joint):
    torch.manual_seed(1)
    original=Bottleneck(12,6).eval().requires_grad_(False)
    fitted=original,torch.randn(12),torch.ones(12)
    model=study.ValueMLP(fitted,hidden=8,joint=joint)
    before={k:v.clone() for k,v in model.state_dict().items()}
    optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.1)
    x=torch.randn(16,12)
    loss,_=study.objective(model,x,torch.arange(16).float()%2,.01)
    loss.backward()
    assert model.head[0].weight.grad is not None
    assert (model.compressor.projection.weight.grad is not None)==joint
    assert model.compressor.readout.weight.grad is None
    assert model.mean.grad is None and model.scale.grad is None
    optimizer.step()
    assert not torch.equal(before['head.0.weight'],model.head[0].weight)
    assert torch.equal(before['compressor.projection.weight'],model.compressor.projection.weight) != joint
    torch.testing.assert_close(original.projection.weight,before['compressor.projection.weight'])
    torch.testing.assert_close(model.mean,before['mean'])
    probability,_=model(x)
    assert ((probability>=0)&(probability<=1)).all()


def test_matched_initialization():
    fitted=Bottleneck(12,6),torch.zeros(12),torch.ones(12)
    torch.manual_seed(42)
    frozen=study.ValueMLP(fitted,8,False)
    torch.manual_seed(42)
    joint=study.ValueMLP(fitted,8,True)
    x=torch.randn(10,12)
    torch.testing.assert_close(frozen(x)[0],joint(x)[0])


def test_end_to_end_and_safe_completed_fit_reuse(tmp_path):
    source,online,out=tmp_path/'features',tmp_path/'online',tmp_path/'out'
    folder=source/'packed/rms_hidden_concat'
    folder.mkdir(parents=True)
    (online/'filters').mkdir(parents=True)
    manifest=dict(model_hashes={'config.json':'fixture'},layers=[27,26,24,20,12],rms_eps=1e-6)
    (source/'manifest.json').write_text(json.dumps(manifest))
    torch.manual_seed(4)
    model=Bottleneck(7680,768)
    path=online/'filters/sparse.pt'
    torch.save(dict(state=model.state_dict(),mean=torch.zeros(7680),scale=torch.ones(7680),width=768),path)
    (online/'manifest.json').write_text(json.dumps(dict(manifest,
        filters={'sparse':dict(path=str(path),sha256=sha256(path))})))
    for i,split in enumerate(('train','val','test')):
        np.save(folder/f'{split}-x.npy',np.random.default_rng(i).normal(size=(6,7680)).astype('float32'))
        torch.save(dict(y=torch.tensor([0.,1.,0.,1.,0.,1.]),question=torch.arange(6)+i*10,
                        response=torch.zeros(6,dtype=torch.long),position=torch.arange(6)),folder/f'{split}-labels.pt')
    (folder/'complete.json').write_text(json.dumps(dict(files={p.name:sha256(p) for p in folder.iterdir()})))
    before={str(p):sha256(p) for root in (source,online) for p in root.rglob('*') if p.is_file()}
    args=SimpleNamespace(features=source,study=online,out=out,hidden=8,seeds=[42],epochs=2,patience=1,
        batch_size=4,head_lr=.001,compressor_lr=.0001,sparsity=.01,weight_decay=.001,ridge_epsilon=.01,device='cpu')
    threads=torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        study.run(args)
        checkpoints={kind:sha256(out/f'{kind}-seed42/best.pt') for kind in study.KINDS}
        study.run(args)
        assert checkpoints=={kind:sha256(out/f'{kind}-seed42/best.pt') for kind in study.KINDS}
        args.hidden=9
        with pytest.raises(ValueError,match='Protocol/code changed'):
            study.run(args)
    finally:
        torch.set_num_threads(threads)
    result=json.loads((out/'summary.json').read_text())
    assert len(result['results']['test'])==3
    assert len(result['paired'])==3
    assert json.loads((out/'status.json').read_text())['state']=='complete'
    assert before=={str(p):sha256(p) for root in (source,online) for p in root.rglob('*') if p.is_file()}
