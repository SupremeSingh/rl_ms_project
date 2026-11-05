"""Offline feature ablation: transformations, paired inputs and Ridge equations."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from slice_critic.bottleneck import Bottleneck
from slice_critic.storage import sha256

SPEC = importlib.util.spec_from_file_location('ridge_normalization',
    Path(__file__).resolve().parents[1]/'tools/diagnostics/ridge_normalization.py')
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def test_transforms_preserve_baseline_and_use_per_example_statistics():
    torch.manual_seed(12)
    model = Bottleneck(6, 4).eval()
    fitted = model, torch.randn(6), torch.ones(6)*2
    x = torch.randn(8, 6)
    baseline = model.features((x-fitted[1])/fitted[2])
    torch.testing.assert_close(study.transform('slice_baseline', x, fitted, 1e-6, 1e-5), baseline)
    normalized = study.transform('slice_layernorm', x, fitted, 1e-6, 1e-5)
    torch.testing.assert_close(normalized, torch.nn.functional.layer_norm(baseline, (4,), eps=1e-5))
    torch.testing.assert_close(normalized[:1], study.transform('slice_layernorm', x[:1], fitted, 1e-6, 1e-5))
    torch.testing.assert_close(normalized.mean(1), torch.zeros(8), atol=1e-6, rtol=0)
    rms = study.transform('rms_two_thirds', x, fitted, 1e-6, 1e-5)
    torch.testing.assert_close(rms, x / torch.sqrt(x.square().mean(1, keepdim=True)+1e-6))
    assert torch.isfinite(study.transform('rms_two_thirds', torch.zeros_like(x), fitted, 1e-6, 1e-5)).all()


def test_ridge_policies_solve_declared_objectives_even_with_dependent_features():
    x = torch.tensor([[1., -1., 1.], [2., -2., 1.], [3., -3., 1.]], dtype=torch.float64)
    y = torch.tensor([0., 1., 1.], dtype=torch.float64)
    a, b = x.T@x, x.T@y
    for policy in study.POLICIES:
        w, checks = study.ridge_fit(a, b, policy, .1)
        eps = .01 if policy == 'legacy' else .1*float(a.trace())/3
        assert checks['epsilon'] == eps
        torch.testing.assert_close((a+eps*torch.eye(3))@w, b, atol=1e-7, rtol=1e-7)
        assert checks['condition_2'] > 1
    for epsilon in (.1, 1., 10., 100.):
        w, checks = study.ridge_fit(a, b, 'fixed', .1, epsilon=epsilon)
        assert checks['epsilon'] == epsilon
        torch.testing.assert_close((a+epsilon*torch.eye(3))@w, b, atol=1e-7, rtol=1e-7)
    with pytest.raises(ValueError,match='positive'):
        study.ridge_fit(a, b, 'fixed', .1, epsilon=-1.)


def test_question_metric_gives_equal_weight_to_questions():
    metrics, grouped = study.evaluate(np.array([0., 0., 1.]), np.array([1., 1., 1.]), np.array([4, 4, 5]))
    assert metrics['raw_brier'] == pytest.approx(2/3)
    assert metrics['question_mean_raw_brier'] == .5
    assert list(grouped) == [1., 0.]


def test_cache_pairing_and_checksums(tmp_path):
    for arm, width in [('rms_hidden_concat',7680), ('hidden_two_thirds',1536)]:
        folder = tmp_path/'packed'/arm
        folder.mkdir(parents=True)
        np.save(folder/'train-x.npy', np.zeros((2,width),dtype=np.float32))
        torch.save(dict(y=torch.tensor([0.,1.]), question=torch.tensor([1,2]),
                        position=torch.tensor([0,1]), response=torch.tensor([0,0])), folder/'train-labels.pt')
        (folder/'complete.json').write_text(json.dumps(dict(files={p.name:sha256(p)
            for p in folder.iterdir()})))
    study.open_split(tmp_path,'train')
    folder=tmp_path/'packed'/'hidden_two_thirds'
    path=folder/'train-labels.pt'
    data=torch.load(path,weights_only=True)
    data['position'][1]=2
    torch.save(data,path)
    with pytest.raises(ValueError,match='checksum'):
        study.open_split(tmp_path,'train')
    complete=json.loads((folder/'complete.json').read_text())
    complete['files'][path.name]=sha256(path)
    (folder/'complete.json').write_text(json.dumps(complete))
    with pytest.raises(ValueError,match='Unpaired'):
        study.open_split(tmp_path,'train')


@pytest.mark.parametrize('extra_epsilons', [[], [.1, 1., 10., 100.]])
def test_complete_offline_run_writes_all_arms_without_modifying_inputs(tmp_path, extra_epsilons):
    source, online, out = tmp_path/'features', tmp_path/'online', tmp_path/'out'
    source.mkdir()
    (online/'filters').mkdir(parents=True)
    layers=[27,26,24,20,12]
    manifest=dict(model_hashes={'config.json':'test-identity'},layers=layers,rms_eps=1e-6)
    (source/'manifest.json').write_text(json.dumps(manifest))
    torch.manual_seed(1)
    model=Bottleneck(7680,768)
    path=online/'filters/sparse.pt'
    torch.save(dict(state=model.state_dict(),mean=torch.zeros(7680),scale=torch.ones(7680),
                    width=768,layers=layers,rms_eps=1e-6),path)
    (online/'manifest.json').write_text(json.dumps(dict(manifest,
        filters={'sparse':dict(path=str(path),sha256=sha256(path))})))
    for arm,width in [('rms_hidden_concat',7680),('hidden_two_thirds',1536)]:
        folder=source/'packed'/arm
        folder.mkdir(parents=True)
        for i,split in enumerate(('train','val','test')):
            np.save(folder/f'{split}-x.npy',np.random.default_rng(i).normal(size=(6,width)).astype('float32'))
            torch.save(dict(y=torch.tensor([0.,1.,0.,1.,0.,1.]),
                            question=torch.tensor([10*i+j//2 for j in range(6)]),
                            response=torch.zeros(6,dtype=torch.long),position=torch.arange(6)),
                       folder/f'{split}-labels.pt')
        (folder/'complete.json').write_text(json.dumps(dict(files={p.name:sha256(p) for p in folder.iterdir()})))
    before={str(p):sha256(p) for root in (source,online) for p in root.rglob('*') if p.is_file()}
    threads=torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        study.run(SimpleNamespace(features=source,study=online,out=out,strength=1e-4,
                                  layernorm_eps=1e-5,batch_size=4,ridge_epsilons=extra_epsilons))
    finally:
        torch.set_num_threads(threads)
    summary=json.loads((out/'summary.json').read_text())
    policies=2+len(extra_epsilons)
    assert len(summary['fits'])==3*policies
    assert len(summary['results']['test'])==3*policies
    assert len(summary['paired'])==2*policies+3*(policies-1)
    assert summary['protocol']['layernorm_eps']==1e-5
    for arm in study.ARMS:
        for epsilon in extra_epsilons:
            key=f'{arm}/fixed_{epsilon:.12g}'
            assert summary['fits'][key]['epsilon']==epsilon
    assert json.loads((out/'status.json').read_text())['state']=='complete'
    assert before=={str(p):sha256(p) for root in (source,online) for p in root.rglob('*') if p.is_file()}
