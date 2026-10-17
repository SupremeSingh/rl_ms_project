import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import export_probe_poster as poster


def put(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def fixture_study(tmp_path):
    source,sparse,out=[tmp_path/name for name in ('source','sparse','export')]
    source.mkdir();sparse.mkdir()
    questions=[dict(index=i,id=f'q{i}',split=split) for i,split in enumerate(('train','val','test'))]
    put(source/'questions.json',questions)
    cfg=dict(width=2,seeds=[42,43],packed={})
    data={}; allmeta={}
    for split,q in zip(('train','val','test'),questions):
        values={k:[] for k in poster.FIELDS}
        rows=[]
        for response in range(2):
            rows.append(dict(score=response,verifier_status='correct' if response else 'incorrect',
                token_ids=list(range(50)),finish_reason='stop'))
            values['y'] += [response]*8
            values['question'] += [q['index']]*8
            values['response'] += [response]*8
            values['position'] += poster.prefix_positions(50,q['index'],response)
        values={k:torch.tensor(v) for k,v in values.items()};allmeta[split]=values
        put(source/'trajectories'/f'{q["index"]:05d}.json',dict(responses=rows))
    for arm in poster.ARMS:
        directory=source/'packed'/arm;directory.mkdir(parents=True)
        record=dict(counts={},files={})
        for split,values in allmeta.items():
            path=directory/f'{split}-labels.pt';torch.save(values,path)
            record['counts'][split]=16;record['files'][path.name]=poster.sha256(path)
        put(directory/'complete.json',record)
        cfg['packed'][arm]=poster.sha256(directory/'complete.json')
    manifest=dict(dimension=3,layers=[2,1],questions_sha256=poster.sha256(source/'questions.json'))
    put(source/'manifest.json',manifest)
    cfg['source_manifest']=poster.sha256(source/'manifest.json');put(sparse/'protocol.json',cfg)
    p=np.array([.2]*8+[.8]*8)
    predictions={f'{arm}/ridge':p for arm in poster.ARMS}
    for kind in ('plain','sparse'):
        for seed in cfg['seeds']:predictions[f'{kind}-{seed}/ridge']=p
    np.savez_compressed(sparse/'predictions.npz',**predictions,
        y=allmeta['test']['y'].numpy(),question=allmeta['test']['question'].numpy())
    put(sparse/'summary.json',dict(results={k:poster.metrics(allmeta['test']['y'].numpy(),v) for k,v in predictions.items()}))
    put(sparse/'screen.json',dict(selected={}))
    put(source/'summary.json',dict(counts={}))
    return source,sparse,out


def test_export_reconstructs_exact_prefixes_and_keeps_sources_unchanged(tmp_path):
    source,sparse,out=fixture_study(tmp_path)
    before={p:poster.sha256(p) for root in (source,sparse) for p in root.rglob('*') if p.is_file()}
    poster.export(source,sparse,out,False)
    record=json.loads((out/'summary.json').read_text())
    assert record['counts']['test']['retained_answers']==2
    assert record['counts']['test']['prefixes']==16
    assert record['results']['sparse_2/ridge']['mean']['accuracy']==1
    assert record['results']['constant_train_prior']['mean']['accuracy']==.5
    assert record['by_prefix']['question_only']['results']['sparse_2/ridge']['prefixes']==2
    assert record['dimensions']['projection_parameters']==14
    assert record['benchmark'] is None
    assert '<svg' in (out/'accuracy.svg').read_text()
    assert all(poster.sha256(path)==checksum for path,checksum in before.items())
    with pytest.raises(ValueError,match='empty'):poster.export(source,sparse,out,False)


def test_export_rejects_misaligned_saved_predictions(tmp_path):
    source,sparse,out=fixture_study(tmp_path)
    path=sparse/'predictions.npz'
    with np.load(path) as old: values={k:old[k] for k in old.files}
    values['question']=values['question']+1
    np.savez_compressed(path,**values)
    with pytest.raises(ValueError,match='unaligned'):poster.export(source,sparse,out,False)


def test_trajectory_verification_detects_future_or_changed_labels(tmp_path):
    source,_,_=fixture_study(tmp_path)
    q=poster.read(source/'questions.json')
    data=poster.metadata(source,'rms_hidden_concat','test')
    data['position'][1]=50 # Terminal prefix was never permitted.
    with pytest.raises(ValueError,match='position'):poster.verify_test_trajectories(source,q,data)


def test_metrics_average_heads_rather_than_ensemble():
    y=np.array([0,1]);p=np.array([[.9,.9],[.1,.1]])
    r=poster.summarize(y,p)
    assert r['mean']['brier']==pytest.approx(.41)
    assert r['mean']['brier']!=pytest.approx(poster.metrics(y,p.mean(0))['brier'])



def test_cpu_benchmark_includes_bottleneck_and_readout(tmp_path):
    from math_rl.sparse_bottleneck import Bottleneck
    source,sparse,_=fixture_study(tmp_path)
    for arm,dim in [('rms_hidden_concat',6),('hidden_two_thirds',3),('final',3)]:
        np.save(source/'packed'/arm/'test-x.npy',np.ones((16,dim),dtype=np.float32))
    selected={}
    for kind in ('plain','sparse'):
        path=sparse/f'{kind}.pt';model=Bottleneck(6,2)
        torch.save(dict(state=model.state_dict(),mean=torch.zeros(6),scale=torch.ones(6)),path)
        selected[kind+'-42']=dict(path=path.name,sha256=poster.sha256(path),validation={'brier':.2})
    result=poster.benchmark(source,sparse,dict(selected=selected),2,batch=8,repeats=2)
    assert len(result['operations'])==5
    assert result['checkpoints']['sparse']['projection_parameters']==14
    assert all(r['median_microseconds_per_prefix']>0 for r in result['operations'].values())
