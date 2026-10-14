import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch
from math_rl.sparse_bottleneck import Bottleneck, encode, loss
from math_rl.large_linear import statistics, accumulate, solve
from math_rl.provenance import sha256


def test_sparsity_is_matched_architecture_with_different_objective():
    torch.manual_seed(19)
    model=Bottleneck(9,3)
    x=torch.randn(12,9);y=torch.arange(12).float()%2
    logits,z=model(x)
    assert z.shape==(12,3) and (z>=0).all()
    torch.testing.assert_close(loss(model,x,y,.01)-loss(model,x,y,0),.01*z.mean())
    loss(model,x,y,.01).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    before=model.features(x).detach()
    with torch.no_grad():model.projection.weight.mul_(3)
    torch.testing.assert_close(model.features(x),before)


def test_frozen_features_work_with_linear_lstd_and_terminal_reward():
    torch.manual_seed(7)
    model=Bottleneck(9,3).eval().requires_grad_(False)
    x=torch.randn(8,9)
    z=encode(model,x,torch.zeros(9),torch.ones(9),'cpu',batch=3)
    torch.testing.assert_close(z,model.features(x))
    stats=statistics(3)
    accumulate(stats,z[:4],1,torch.zeros(3),torch.ones(3))
    accumulate(stats,z[4:],0,torch.zeros(3),torch.ones(3))
    for head in ('ridge','lstd-0.99'):
        w,checks=solve(stats,head)
        assert torch.isfinite(w).all() and checks['relative_residual']<1e-8
    assert all(p.grad is None for p in model.parameters())


def test_screen_never_loads_test_and_saves_frozen_selection(tmp_path):
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'scripts'))
    spec=importlib.util.spec_from_file_location('sparse_study',root/'scripts/sparse_critic_study.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    source=tmp_path/'source';out=tmp_path/'out';out.mkdir()
    torch.manual_seed(8)
    # Deliberately no test file: screening must use training/validation only.
    for arm in ('rms_hidden_concat','hidden_two_thirds'):
        dest=source/'packed'/arm;dest.mkdir(parents=True)
        hashes={}
        for split in ('train','val'):
            x=torch.randn(24,8);labels=dict(y=(x[:,0]>0).float(),question=torch.arange(24),
                response=torch.arange(24),position=torch.zeros(24,dtype=torch.long))
            np.save(dest/f'{split}-x.npy',x.numpy());torch.save(labels,dest/f'{split}-labels.pt')
            for suffix in ('x.npy','labels.pt'):
                p=dest/f'{split}-{suffix}';hashes[p.name]=sha256(p)
        (dest/'complete.json').write_text(json.dumps(dict(files=hashes)))
    cfg=dict(width=3,epochs=2,seeds=[42],lrs=[.001],sparse_penalty=.01,batch=12,epsilon=.01)
    mod.screen(source,out,{},cfg,torch.device('cpu'))
    result=json.loads((out/'screen.json').read_text())
    assert set(result['selected'])=={'plain-42','sparse-42'}
    for v in result['selected'].values():
        record=mod.load(out/v['path'])
        assert len(record['history'])==2 and record['width']==3
        assert not any(p.requires_grad for p in mod.model_from(record,'cpu').parameters())
    mod.screen(source,out,{},cfg,torch.device('cpu'))
    # Exercise the full report with synthetic all-token statistics and baselines.
    test_labels=dict(y=torch.tensor([0.,1.]*12),question=torch.arange(24)//2,
                     response=torch.arange(24),position=torch.zeros(24,dtype=torch.long))
    source_stats={};norms={};inputs={}
    for arm in ('rms_hidden_concat','hidden_two_thirds','final'):
        dest=source/'packed'/arm;dest.mkdir(parents=True,exist_ok=True)
        x=torch.randn(24,8);inputs[arm]=x
        np.save(dest/'test-x.npy',x.numpy());torch.save(test_labels,dest/'test-labels.pt')
        record=json.loads((dest/'complete.json').read_text()) if (dest/'complete.json').exists() else {'files':{}}
        for suffix in ('x.npy','labels.pt'):
            p=dest/f'test-{suffix}';record['files'][p.name]=sha256(p)
        (dest/'complete.json').write_text(json.dumps(record))
        s=statistics(8)
        for i in range(24): accumulate(s,x[i:i+1],int(test_labels['y'][i]),torch.zeros(8),torch.ones(8))
        source_stats[arm]=s;norms[arm]=(torch.zeros(8),torch.ones(8))
        selected={};fits=source/'linear-fits'/arm;fits.mkdir(parents=True)
        for head in ('ridge','lstd-0.99'):
            w,_=solve(s,head);path=fits/f'{head}.pt'
            torch.save(dict(weights=w,mean=torch.zeros(8),scale=torch.ones(8)),path)
            selected[head]=dict(path=str(path.relative_to(source)),sha256=sha256(path))
        (fits/'selection.json').write_text(json.dumps(dict(selected=selected)))
    folder=source/'linear-statistics';folder.mkdir()
    path=folder/'shard-0.pt';torch.save(dict(stats=source_stats,normalization=norms),path)
    path.with_suffix('.json').write_text(json.dumps(dict(sha256=sha256(path))))
    totals={}
    for k,v in result['selected'].items():
        r=mod.load(out/v['path']);z=encode(mod.model_from(r,'cpu'),inputs['rms_hidden_concat'],r['mean'],r['scale'],'cpu')
        totals[k]=statistics(3)
        for i in range(24):accumulate(totals[k],z[i:i+1],int(test_labels['y'][i]),torch.zeros(3),torch.ones(3))
    (out/'statistics').mkdir()
    mod.save(out/'statistics/shard-0.pt',dict(complete=True,stats=totals,
        lock={k:v['sha256'] for k,v in result['selected'].items()}))
    mod.report(source,out,{'config':{'shards':1}},cfg,torch.device('cpu'))
    final=json.loads((out/'summary.json').read_text())
    assert len(final['results'])==16
    assert 'sparse/ridge_minus_plain' in final['comparisons']
