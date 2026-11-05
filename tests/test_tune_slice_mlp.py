"""Test head-first initialization, selection integrity, and no test access in tuning."""
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
    spec=importlib.util.spec_from_file_location('tune_slice_mlp',TOOLS/'tune_slice_mlp.py')
    tune=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=tune  # dataclasses resolves the declaring module
    spec.loader.exec_module(tune)
    from slice_mlp import ValueMLP,train_one
finally:
    sys.path.remove(str(TOOLS))


def test_warm_checkpoint_can_win_without_any_finetuning(tmp_path):
    torch.manual_seed(1)
    fitted=Bottleneck(12,6),torch.zeros(12),torch.ones(12)
    model=ValueMLP(fitted,8)
    warm=tmp_path/'warm.pt'
    torch.save(dict(state=model.state_dict()),warm)
    data=np.random.default_rng(1).normal(size=(8,12)).astype('float32'),{'y':torch.tensor([0.,1.]*4)}
    args=SimpleNamespace(device='cpu',hidden=8,head_lr=0.,compressor_lr=0.,weight_decay=0.,
        epochs=2,patience=1,batch_size=4,sparsity=0.)
    result=train_one(fitted,data,data,args,'joint_mlp',42,tmp_path,initial_checkpoint=warm)
    assert result['best_epoch']==0
    assert result['initial_checkpoint_sha256']==sha256(warm)
    checkpoint=torch.load(tmp_path/'joint_mlp-seed42/best.pt',weights_only=True)
    for key,value in model.state_dict().items():
        torch.testing.assert_close(checkpoint['state'][key],value)


def test_validation_only_then_locked_evaluation_and_tamper_rejection(tmp_path,monkeypatch):
    source,online,out=tmp_path/'features',tmp_path/'online',tmp_path/'out'
    folder=source/'packed/rms_hidden_concat'
    folder.mkdir(parents=True)
    (online/'filters').mkdir(parents=True)
    manifest=dict(model_hashes={'config.json':'fixture'},layers=[27,26,24,20,12],rms_eps=1e-6)
    (source/'manifest.json').write_text(json.dumps(manifest))
    (folder/'complete.json').write_text('{}')
    model=Bottleneck(7680,768)
    path=online/'filters/sparse.pt'
    torch.save(dict(state=model.state_dict(),mean=torch.zeros(7680),scale=torch.ones(7680),width=768),path)
    (online/'manifest.json').write_text(json.dumps(dict(manifest,
        filters={'sparse':dict(path=str(path),sha256=sha256(path))})))
    recipe=tune.Recipe(head_lrs=(1e-3,),compressor_lrs=(1e-5,),penalties=(0.,),seeds=(42,43))
    monkeypatch.setattr(tune,'Recipe',lambda:recipe)
    calls=[]
    def cached(source,split):
        calls.append(split)
        i=('train','val','test').index(split)
        return np.random.default_rng(i).normal(size=(6,7680)).astype('float32'),dict(
            y=torch.tensor([0.,1.]*3),question=torch.arange(6)+10*i,
            response=torch.zeros(6,dtype=torch.long),position=torch.arange(6))
    monkeypatch.setattr(tune,'cached_split',cached)
    args=SimpleNamespace(features=source,study=online,out=out,device='cpu',epochs=1,
                         patience=1,batch_size=4,stage='tune')
    threads=torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        tune.run(args)
        assert 'test' not in calls
        assert not (out/'summary.json').exists()
        selection=tune.verify_selection(out)
        assert len(selection['models']['joint_mlp'])==2
        assert json.loads((out/'status.json').read_text())['state']=='validation_complete'
        lock=sha256(out/'selection.json')
        args.stage='evaluate'
        tune.run(args)
        assert calls.count('test')==1
        assert sha256(out/'selection.json')==lock
        summary=json.loads((out/'summary.json').read_text())
        assert len(summary['results'])==6  # two Ridge variants + 2 models x 2 seeds
        assert len(summary['paired'])==5
        assert json.loads((out/'status.json').read_text())['state']=='complete'
        (out/'selection.json').write_text('{}')
        with pytest.raises(ValueError,match='changed'):
            tune.verify_selection(out)
    finally:
        torch.set_num_threads(threads)
