"""Real tiny-Qwen attention/gradient tests plus offline data and submission contracts."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import subprocess

import numpy as np
import pytest
import torch
from tools.encoder import model as enc
from tools.encoder import study
from slice_critic.storage import save_tensor,write_json,sha256


def tiny():
    from transformers import Qwen2Config,Qwen2ForCausalLM
    torch.manual_seed(9)
    return Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=24,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,attention_dropout=0.,
        _attn_implementation='eager')).eval()


def test_attention_is_actually_bidirectional_without_touching_native_causal_path():
    model=tiny();probe=enc.attention_check(model,'cpu')
    assert probe['causal_earlier_token_change']<=1e-5
    assert probe['bidirectional_earlier_token_change']>1e-6
    assert model.training is False


def test_decoder_single_pass_matches_truncated_prefixes():
    model=tiny();prompt=[1,2];answer=[3,4,5];positions=[0,1,2]
    packed=enc.causal_features(model,prompt,answer,positions,[1,0],0,'cpu')
    for i,p in enumerate(positions):
        prefix=enc.prefix_features(model,prompt+answer[:p],[1,0],0,False,'cpu')
        for key in packed:torch.testing.assert_close(packed[key][i],prefix[key],atol=2e-6,rtol=2e-5)
    # No answer suffix is accepted by the bidirectional prefix extractor.
    prefix=prompt+answer[:1]
    first=enc.prefix_features(model,prefix,[1,0],0,True,'cpu')
    future_changed=[3,19,20]
    second=enc.prefix_features(model,prompt+future_changed[:1],[1,0],0,True,'cpu')
    torch.testing.assert_close(first['concat'],second['concat'])


def test_mntp_shifts_targets_and_masks_all_selected_inputs(monkeypatch):
    seen=[]
    def forward(model,ids,bidirectional):
        seen.append(ids.clone());assert bidirectional
        return torch.eye(4).unsqueeze(0)
    monkeypatch.setattr(enc,'forward',forward)
    model=SimpleNamespace(lm_head=torch.nn.Identity(),config=SimpleNamespace(vocab_size=4))
    ids=torch.tensor([[0,1,2,3]])
    actual=enc.masked_next_loss(model,ids,0,torch.Generator().manual_seed(9),probability=1.)
    expected=torch.nn.functional.cross_entropy(torch.eye(4)[:3],torch.tensor([1,2,3]))
    torch.testing.assert_close(actual,expected)
    assert seen[0].tolist()==[[0,0,0,0]] and ids.tolist()==[[0,1,2,3]]
    with pytest.raises(ValueError):enc.masked_next_loss(model,ids[:,:1],0,torch.Generator())


def test_only_adapters_update_and_can_be_restored(tmp_path):
    model=tiny();enc.install_adapters(model,2)
    model.train();model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    assert all(p.dtype==torch.float32 for p in model.parameters() if p.requires_grad)
    loss=enc.masked_next_loss(model,torch.tensor([[1,2,3,4]]),0,torch.Generator().manual_seed(12))
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters() if n.endswith('.b'))
    assert all(p.grad is None for n,p in model.named_parameters() if not n.endswith(('.a','.b')))
    path=tmp_path/'adapter.pt';save_tensor(path,dict(weights=enc.adapter_state(model),rank=2))
    restored=tiny();enc.restore_adapter(restored,path)
    for key,value in enc.adapter_state(model).items():torch.testing.assert_close(value,enc.adapter_state(restored)[key])


def test_contrastive_views_vary_and_gradients_flow():
    model=tiny();enc.install_adapters(model,2);model.train()
    for layer in model.model.layers:layer.self_attn.attention_dropout=.3
    ids=[torch.tensor([[1,2,3,4]]),torch.tensor([[5,6,7,8]])]
    left=torch.cat([enc.embedding(model,x) for x in ids]);right=torch.cat([enc.embedding(model,x) for x in ids])
    assert not torch.equal(left,right)
    objective=enc.contrastive_loss(left,right);objective.backward()
    assert torch.isfinite(objective)
    assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters() if n.endswith('.b'))


def test_adaptation_uses_train_only_and_never_outcome_labels(monkeypatch):
    rows=[]
    for split in ('train','val','test'):
        for answer in range(2):
            rows.append((dict(id=split,split=split),answer,[1,2],dict(token_ids=[3,4,5,6])))
    monkeypatch.setattr(study,'records',lambda *a:iter(rows))
    assert study.adaptation_corpus(None,dict(adapt_length=512))==[('train',[1,2,3,4])]


def test_shared_evaluation_bank_but_distinct_training_units(tmp_path):
    write_json(tmp_path/'manifest.json',{})
    folder=tmp_path/'features/decoder';folder.mkdir(parents=True)
    for split in ('train','val','test'):
        x=torch.arange(12).reshape(3,4).float()
        save_tensor(folder/f'{split}.pt',dict(split=split,positions=[0,1,2],
            units=dict(token=[0,1],chunk=[0,2]),concat=x,raw=x,rms=x,y=1.,question_id=split))
    write_json(folder/'complete.json',dict(manifest_sha256=sha256(tmp_path/'manifest.json'),
        files={p.name:sha256(p) for p in folder.glob('*.pt')}))
    for split in ('val','test'):
        a=study.collection(tmp_path,'decoder','token',split);b=study.collection(tmp_path,'decoder','chunk',split)
        torch.testing.assert_close(a['raw'],b['raw']);assert len(a['y'])==3
    a=study.collection(tmp_path,'decoder','token','train');b=study.collection(tmp_path,'decoder','chunk','train')
    assert not torch.equal(a['raw'],b['raw'])


def controller():
    path=Path(__file__).resolve().parents[1]/'scripts/submit_encoder_study.py'
    spec=importlib.util.spec_from_file_location('submit_encoder',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_scheduler_chain_can_avoid_cpu_partition(tmp_path,monkeypatch):
    m=controller();source=tmp_path/'source';source.mkdir();write_json(source/'manifest.json',{})
    calls=[]
    def run(args,**kwargs):calls.append(args);return SimpleNamespace(returncode=0,stdout=str(len(calls)),stderr='')
    monkeypatch.setattr(m.subprocess,'run',run)
    m.submit(source,tmp_path/'out',cpu_on_gpu=True)
    assert len(calls)==7 and all('--gres=gpu:a5000:1' in row for row in calls)
    assert '--array=0-2%3' in calls[4] and '--dependency=afterok:4' in calls[4]
    assert calls[0][-1]=='0'


def test_scheduler_timeout_does_not_duplicate_submission(tmp_path,monkeypatch):
    m=controller();source=tmp_path/'source';source.mkdir();write_json(source/'manifest.json',{})
    calls=[]
    def run(args,**kwargs):calls.append(args);raise subprocess.TimeoutExpired(args,90)
    monkeypatch.setattr(m.subprocess,'run',run)
    with pytest.raises(RuntimeError,match='Check Slurm'):m.submit(source,tmp_path/'out')
    with pytest.raises(FileExistsError):m.submit(source,tmp_path/'out')
    assert len(calls)==1
