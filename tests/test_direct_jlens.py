"""Complete-score equivalence, ridge equivalence and selection isolation."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from math_rl.direct_jlens import compact_readout, isotropic_normalization, readout_factor, rms_coordinates


@pytest.mark.parametrize('rank_deficient',[False,True])
def test_complete_vocabulary_gram_and_ridge_predictions(rank_deficient):
    torch.manual_seed(31)
    U=torch.randn(19,5,dtype=torch.float64)
    if rank_deficient:U[:,-1]=0
    gain=torch.rand(5,dtype=torch.float64)+.1
    factor=readout_factor(U,gain,chunk_size=7)
    h=torch.randn(16,5,dtype=torch.float64)
    compact=compact_readout(h,factor['factor'],1e-6)
    full=(rms_coordinates(h,1e-6)*gain)@U.T
    torch.testing.assert_close(compact@compact.T,full@full.T,rtol=1e-10,atol=1e-10)
    # Penalized linear fits on all logits vs lossless coordinates agree on new rows.
    y=torch.randn(12,dtype=torch.float64)
    def prediction(x):
        train=x[:12];test=x[12:]
        mean=train.mean(0);target=y-y.mean();train=train-mean
        w=torch.linalg.solve(train.T@train+.2*torch.eye(x.shape[1],dtype=x.dtype),train.T@target)
        return (test-mean)@w+y.mean()
    torch.testing.assert_close(prediction(compact),prediction(full),rtol=1e-8,atol=1e-8)
    assert factor['truncation']=='none'


def test_isotropic_scale_preserves_geometry_and_needs_training_only():
    x=torch.tensor([[1.,2.,9.],[3.,6.,5.],[7.,2.,2.]])
    mean,scale=isotropic_normalization(x)
    assert scale.unique().numel()==1
    torch.testing.assert_close(mean,x.mean(0))
    expected=(x-mean)@(x-mean).T/scale[0]**2
    torch.testing.assert_close(((x-mean)/scale)@((x-mean)/scale).T,expected)


def load_module(monkeypatch):
    scripts=Path(__file__).resolve().parents[1]/'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec=importlib.util.spec_from_file_location('direct_script',scripts/'compare_direct_jlens.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_fit_selects_all_alphas_before_test(tmp_path,monkeypatch):
    m=load_module(monkeypatch)
    torch.manual_seed(91)
    for split,start in [('train',0),('val',10),('test',20)]:
        metadata=dict(y=torch.arange(24).float()%2,question=torch.arange(start,start+4).repeat_interleave(6),
                      response=torch.arange(24),position=torch.ones(24,dtype=torch.long))
        for arm in m.ARMS:
            (tmp_path/arm).mkdir(exist_ok=True)
            torch.save(dict(metadata,x=torch.randn(24,4 if arm=='hidden_two_thirds' else 12)),tmp_path/arm/f'{split}.pt')
    load=torch.load
    def checked(path,*a,**kw):
        if Path(path).name=='test.pt':assert (tmp_path/'selection.json').exists()
        return load(path,*a,**kw)
    monkeypatch.setattr(torch,'load',checked)
    m.fit_and_report(tmp_path,{'layers':[4,3,1],'dimension':4})
    summary=json.loads((tmp_path/'summary.json').read_text())
    assert len(summary['results'])==4 and len(summary['paired'])==3


def test_cached_run_without_sparse_artifacts(tmp_path,monkeypatch):
    import sys
    m=load_module(monkeypatch)
    from math_rl.provenance import sha256
    source,model,out=[tmp_path/n for n in ('source','model','out')]
    source.mkdir();model.mkdir()
    (source/'status.json').write_text('{"state":"complete"}')
    (model/'config.json').write_text('{"rms_norm_eps":1e-6}')
    (source/'lens.pt').write_bytes(b'fixture')
    (source/'lens-metadata.json').write_text(json.dumps({'sha256':sha256(source/'lens.pt')}))
    manifest=dict(environment={'test':'1'},inputs={},layers=[4,3,1],target_layer=4,dimension=4)
    (source/'manifest.json').write_text(json.dumps(manifest))
    files={}
    for split in ('train','val','test'):
        metadata=dict(y=torch.arange(24).float()%2,question=torch.arange(4).repeat_interleave(6),
                      response=torch.arange(24),position=torch.ones(24,dtype=torch.long))
        for name in ('hidden_two_thirds','hidden-block5','hidden-block4','hidden-block2'):
            folder=source/name;folder.mkdir(exist_ok=True)
            path=folder/f'{split}.pt';torch.save(dict(metadata,x=torch.randn(24,4)),path)
            files[str(path.relative_to(source))]=sha256(path)
    (source/'extraction.json').write_text(json.dumps({'files':files}))
    lens=SimpleNamespace(transport=lambda h,l:h*1.1)
    monkeypatch.setitem(sys.modules,'jlens',SimpleNamespace(JacobianLens=SimpleNamespace(load=lambda p:lens)))
    monkeypatch.setattr(m.original,'MODEL',model)
    monkeypatch.setattr(m.original,'installed_lens',lambda:{'test':'1'})
    monkeypatch.setattr(m,'load_readout_weights',lambda p:{'lm_head.weight':torch.randn(10,4),'model.norm.weight':torch.ones(4)})
    monkeypatch.delenv('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE',raising=False)
    m.run(SimpleNamespace(source=source,out=out,batch_size=8,device='cpu'))
    assert json.loads((out/'status.json').read_text())['state']=='complete'
    assert len(json.loads((out/'readout-checks.json').read_text()))==9
    assert torch.load(out/'direct_jlens_concat/train.pt',weights_only=True)['x'].shape==(24,12)
