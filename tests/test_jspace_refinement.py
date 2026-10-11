"""Pursuit numerical checks and cached-feature refinement integration."""
import importlib.util
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import torch

from math_rl.jspace import pursuit_path, reconstruct


def module():
    scripts = Path(__file__).resolve().parents[1]/'scripts'
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location('refinement', scripts/'refine_jspace_probes.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_matching_path_reproduces_original():
    torch.manual_seed(2)
    dictionary = torch.nn.functional.normalize(torch.randn(12, 5), dim=1)
    h = torch.randn(7, 5)
    for k, result in pursuit_path(h,dictionary,[2,4,6],'matching').items():
        for actual, expected in zip(result,reconstruct(h,dictionary,k)):
            torch.testing.assert_close(actual,expected)


def test_gradient_refits_correlated_atoms_and_handles_zero():
    dictionary = torch.tensor([[1.,0.],[.6,.8],[-1.,0.],[0.,-1.]])
    h = torch.tensor([[1.6,.8],[0.,0.]])
    result, ids, weights = pursuit_path(h,dictionary,[2],'gradient',refinements=100)[2]
    greedy = reconstruct(h,dictionary,2)[0]
    assert ((h-result)**2).sum() < ((h-greedy)**2).sum() * .01
    assert (weights>=0).all()
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result[1],h[1])
    assert all(len(row.unique())==2 for row in ids)


def test_gradient_path_error_is_monotonic():
    torch.manual_seed(29)
    dictionary=torch.nn.functional.normalize(torch.randn(20,8),dim=1)
    h=torch.randn(10,8)
    last=h.square().sum(1)
    for k,(value,ids,weights) in pursuit_path(h,dictionary,list(range(1,10))).items():
        error=(h-value).square().sum(1)
        assert (error<=last+1e-5).all()
        torch.testing.assert_close(value,torch.einsum('bk,bkd->bd',weights,dictionary[ids]))
        last=error
    with pytest.raises(ValueError):pursuit_path(h,dictionary,[21])


def test_subset_outcome_independent_and_keeps_paired_prefixes():
    m=module()
    data=dict(x=torch.randn(24,4),y=torch.zeros(24),question=torch.arange(4).repeat_interleave(6),
              response=torch.tensor([0,0,1,1,2,2]*4),position=torch.tensor([0,3]*12))
    selected=m.subset(data,2,2)
    changed=m.subset(dict(data,y=torch.ones(24)),2,2)
    assert len(selected['y'])==8
    torch.testing.assert_close(selected['x'],changed['x'])
    assert selected['response'].max()==1


def test_cached_reconstruction_and_ridge_pipeline(tmp_path):
    m=module()
    torch.manual_seed(2)
    dictionary=torch.nn.functional.normalize(torch.randn(10,4),dim=1)
    data=dict(x=torch.randn(48,4),y=torch.arange(48).float()%2,
        question=torch.arange(8).repeat_interleave(6),response=torch.tensor([0,0,1,1,2,2]*8),
        position=torch.tensor([0,3]*24))
    train={k:v[:30] for k,v in data.items()};val={k:v[30:] for k,v in data.items()}
    for name in ('features','diagnostics'):(tmp_path/name).mkdir()
    class Tokens:
        def decode(self,ids):return str(ids)
    outputs={split:m.reconstruct_split(tmp_path,split,d,dictionary,['matching','gradient'],[2,3],7,Tokens())
             for split,d in [('train',train),('val',val)]}
    cached=m.reconstruct_split(tmp_path,'train',train,dictionary,['matching','gradient'],[2,3],7,Tokens())
    for name in cached:torch.testing.assert_close(cached[name],outputs['train'][name])
    for arm in ('jspace','jspace_final','random','random_final'):
        control='random' if arm.startswith('random') else 'jspace'
        transformed=[m.features(d,d,outputs[s][f'gradient-3-{control}'],arm)
                     for s,d in [('train',train),('val',val)]]
        fitted=m.fit_head(tmp_path/'heads'/arm,*transformed)
        prediction=m.prediction(fitted,transformed[1])
        assert np.isfinite(prediction).all()
        result=m.paired(prediction,prediction,val['y'].numpy(),val['question'].numpy())
        assert result['mean']==0 and result['interval_95']==[0.,0.]


def test_runner_locks_selection_before_reading_test(tmp_path, monkeypatch):
    m=module()
    source=tmp_path/'source';source.mkdir()
    model=tmp_path/'model';model.mkdir()
    (model/'weights.safetensors').write_bytes(b'fixture')
    (source/'lens.pt').write_bytes(b'fixture')
    extraction={};questions=[];source_hashes={}
    (source/'trajectories').mkdir()
    torch.manual_seed(71)
    for offset,split in enumerate(('train','val','test')):
        data=dict(x=torch.randn(24,4),y=torch.arange(24).float()%2,
            question=torch.arange(offset*4,offset*4+4).repeat_interleave(6),
            response=torch.tensor([0,0,1,1,2,2]*4),position=torch.tensor([0,3]*12))
        for qid in data['question'].unique().tolist():
            questions.append(dict(original_index=qid,id=f'q{qid}',question='A question'))
            path=source/'trajectories'/f'{qid:05d}.json'
            path.write_text(json.dumps(dict(responses=[dict(response='An answer')]*3)))
            source_hashes[str(path)]=m.sha256(path)
        for rep in ('two_thirds','final'):
            (source/rep).mkdir(exist_ok=True)
            path=source/rep/f'{split}.pt'
            torch.save(data,path);extraction[str(path.relative_to(source))]=m.sha256(path)
    for rep in ('two_thirds','final','jspace','jspace_final'):
        (source/rep).mkdir(exist_ok=True)
        (source/rep/'summary.json').write_text('{"heads": []}')
    (source/'manifest.json').write_text(json.dumps(dict(environment={},model_hashes={},
        source=str(source),questions=questions,source_hashes=source_hashes)))
    (source/'extraction.json').write_text(json.dumps(dict(files=extraction)))
    (source/'lens-metadata.json').write_text(json.dumps(dict(sha256=m.sha256(source/'lens.pt'),source_layer_zero_based=2)))
    monkeypatch.setattr(m.original,'MODEL',model)
    monkeypatch.setattr(m.original,'installed_lens',lambda:{})
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.Tensor,'cuda',lambda self:self)
    monkeypatch.delenv('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE',raising=False)
    class Weights:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def keys(self):return ['lm_head.weight','model.norm.weight']
        def get_tensor(self,key):return torch.ones(4) if key=='model.norm.weight' else torch.randn(10,4)
    monkeypatch.setitem(sys.modules,'safetensors',SimpleNamespace(safe_open=lambda *a,**k:Weights()))
    monkeypatch.setitem(sys.modules,'jlens',SimpleNamespace(JacobianLens=SimpleNamespace(
        load=lambda path:SimpleNamespace(jacobians={2:torch.eye(4)}))))
    tokens=SimpleNamespace(decode=lambda ids:str(ids))
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(AutoTokenizer=SimpleNamespace(
        from_pretrained=lambda *a,**k:tokens)))
    out=tmp_path/'out'
    real_load=torch.load
    def checked_load(path,*args,**kwargs):
        if Path(path).name=='test.pt':assert (out/'selection.json').exists()
        return real_load(path,*args,**kwargs)
    monkeypatch.setattr(torch,'load',checked_load)
    monkeypatch.setattr(sys,'argv',['refine',str(source),'--out',str(out),'--smoke'])
    m.main()
    assert json.loads((out/'status.json').read_text())['state']=='complete'
    result=json.loads((out/'summary.json').read_text())
    assert len(result['results'])==7
    assert (out/'responses.html').exists()
