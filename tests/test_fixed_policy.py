"""CPU contracts for shared-policy labels/references; never submit real jobs."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import numpy as np
import pytest
import torch

from math_rl import fixed_policy as study
from math_rl import fixed_policy_gpu as gpu
from math_rl.fixed_policy_neural import preaction_values, return_loss, equal_question_mse
from math_rl.provenance import sha256,write_json

ROOT=Path(__file__).resolve().parents[1]


def test_causal_target_alignment_and_pooled_token_gradients():
    x=torch.arange(6,dtype=torch.float32,requires_grad=True)
    torch.testing.assert_close(preaction_values(x,2,4),torch.tensor([1.,2.,3.,4.]))
    loss=return_loss(x,2,4,1,4)
    loss.backward()
    assert x.grad[0]==0 and x.grad[-1]==0
    a=torch.tensor([0.,.2,.5,.8],requires_grad=True)
    b=torch.tensor([0.,.9],requires_grad=True)
    # Simulate FSDP's average of two rank gradients on unequal answer lengths.
    distributed=(return_loss(a,1,3,1,4,2)+return_loss(b,1,1,0,4,2))/2
    expected=((a[:3]-1).square().sum()+b[:1].square().sum())/4
    torch.testing.assert_close(distributed,expected)


def test_validation_epoch_selection_weights_questions_equally():
    assert equal_question_mse([('a',1.),('b',0.),('b',0.),('b',0.)])==.5
    with pytest.raises(ValueError):equal_question_mse([])


def test_linear_arms_share_train_answers_and_ignore_val_test_labels(tmp_path,monkeypatch):
    rows=[dict(key='train',split='train'),dict(key='val',split='val'),dict(key='test',split='test')]
    seen=[]
    monkeypatch.setattr(study,'items',lambda *a:rows)
    monkeypatch.setattr(study,'answer',lambda out,row:seen.append(row['key']) or dict(score=1 if row['split']=='train' else 0))
    f=dict(slice=torch.tensor([[1.,0.],[2.,1.],[3.,1.]]),rms=torch.tensor([[2.,0.],[3.,1.],[4.,2.]]),starts=[0,2])
    monkeypatch.setattr(study,'features',lambda *a:f)
    write_json(tmp_path/'manifest.json',{})
    study.linear_fit(tmp_path,dict(epsilon=.01))
    assert seen==['train']
    heads=torch.load(tmp_path/'ridge.pt',weights_only=True)
    assert heads['slice-token']['states']==3 and heads['slice-chunk']['states']==2
    for arm in study.ARMS[:4]:
        x=f['slice' if arm.startswith('slice') else 'rms']
        if arm.endswith('chunk'):x=x[f['starts']]
        x=study.design(x)
        expected=torch.linalg.solve(x.T@x+.01*torch.eye(3),x.sum(0))
        torch.testing.assert_close(heads[arm]['weights'],expected)


def test_bank_selects_test_only_preserves_actions_and_horizon(tmp_path,monkeypatch):
    rows=[dict(key='a',split='train',id='train',ground_truth='1'),
          dict(key='b',split='test',id='test',ground_truth='1')]
    monkeypatch.setattr(study,'items',lambda *a:rows)
    monkeypatch.setattr(study,'answer',lambda *a:dict(prompt_ids=[1,2],token_ids=[999,3,4,5,2]))
    monkeypatch.setattr(study,'features',lambda *a:dict(starts=[0,2,4]))
    write_json(tmp_path/'manifest.json',{})
    study.pack(tmp_path,dict(prefixes=4,response_cap=2048))
    states=study.read(tmp_path/'bank.json')['states']
    assert len(states)==3 and {s['question_id'] for s in states}=={'test'}
    assert states[1]['prefix_ids']==[999,3] and states[1]['remaining_tokens']==2046
    assert states[0]['prefix_ids']==[]


def test_mc_uses_one_actor_and_one_reference_bank_not_six(tmp_path,monkeypatch):
    m=dict(actor_path='one-actor',shards=4,seed=92017,recipe=dict(samples=4),reference_batch=16,response_cap=2048)
    write_json(tmp_path/'manifest.json',{})
    states=[dict(state_id=f'q/{i}',question_id='q',ground_truth='1',prompt_ids=[1,2],
                 prefix_ids=[9]*i,remaining_tokens=2048-i) for i in range(2)]
    write_json(tmp_path/'bank.json',dict(states=states))
    calls=[];actors=[]
    class Generator:
        def __init__(self,actor,m):actors.append(str(actor))
        def generate(self,prompt,count,remaining,seed):
            calls.append((prompt,count,remaining,seed))
            return [dict(token_ids=[3,2],score=i%2,verifier_status='correct' if i%2 else 'incorrect',capped=False) for i in range(count)]
    monkeypatch.setattr(gpu,'Generator',Generator)
    monkeypatch.setattr(gpu,'grade',lambda g,gold,generated,prefix:generated)
    for task in range(4):gpu.mc(tmp_path,m,task)
    assert set(actors)=={'one-actor'}
    assert sum(r[1] for r in calls)==12  # 2 states * 4 references + one 4-answer group.
    assert len({r[3] for r in calls})==3
    p=study.read(tmp_path/'predictions-grpo.json')['states']
    assert [r['prediction'] for r in p]==[.5,.5]
    assert study.read(tmp_path/'reference-1.json')['states'][0]['successes']==2
    n=len(calls)
    gpu.mc(tmp_path,m,0)
    assert len(calls)==n


def test_paired_comparisons_cancel_shared_reference_errors():
    rows=[dict(question_id=str(i),samples=128,successes=i*32,
               predictions={a:.5 for a in study.ARMS}) for i in range(4)]
    for value in study.paired(rows).values():
        assert value['mean']==0 and value['question_bootstrap_95']==[0.,0.]


def test_report_panels_have_identical_reference_values(tmp_path):
    pytest.importorskip('matplotlib')
    write_json(tmp_path/'manifest.json',{})
    states=[dict(state_id=f'q{i}/0',question_id=f'q{i}',progress=i*.5) for i in range(2)]
    write_json(tmp_path/'bank.json',dict(states=states))
    for task in range(4):
        rows=[dict(state_id=s['state_id'],samples=4,successes=i*2+1) for i,s in enumerate(states) if i%4==task]
        write_json(tmp_path/f'reference-{task}.json',study.output(tmp_path,rows))
    for j,arm in enumerate(study.ARMS):
        rows=[dict(state_id=s['state_id'],prediction=-.1+j*.1+i*.2) for i,s in enumerate(states)]
        write_json(tmp_path/f'predictions-{arm}.json',study.output(tmp_path,rows))
    for name in ('ridge.pt','ridge-statistics.pt','linear.json','neural.json'):(tmp_path/name).write_text('{}')
    study.report(tmp_path,dict(recipe=dict(samples=4),protocol='fixed-policy-critics-v1',scope='Synthetic CPU fixture.'))
    result=study.read(tmp_path/'summary.json')
    assert all([r['reference'] for r in result['panels'][arm]]==[.25,.75] for arm in study.ARMS)
    assert result['panels']['slice-token'][0]['prediction']==-.1  # No primary clipping.
    assert (tmp_path/'value-scatter.pdf').read_bytes().startswith(b'%PDF')
    assert study.read(tmp_path/'verification.json')['passed']
    assert study.read(tmp_path/'progress-counts.json')['0']['questions']==1


def controller():
    spec=importlib.util.spec_from_file_location('submit_fixed_policy',ROOT/'scripts/submit_fixed_policy.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_scheduler_uses_gpu_partition_for_every_stage_and_two_gpus_for_neural(tmp_path,monkeypatch):
    c=controller();source=tmp_path/'source';write_json(source/'pilot/verification.json',{})
    calls=[]
    def run(args,**kw):calls.append(args);return SimpleNamespace(returncode=0,stdout=str(len(calls)),stderr='')
    monkeypatch.setattr(c.subprocess,'run',run)
    c.submit(source,tmp_path/'out','smoke',concurrency=3)
    assert len(calls)==9
    assert all('--partition=compsci-gpu' in row for row in calls)
    assert '--gres=gpu:a5000:2' in calls[5]
    assert '--array=0-5%3' in calls[6] and '--array=0-3%3' in calls[7]
    for i,row in enumerate(calls[1:],1):assert f'--dependency=afterok:{i}' in row


def test_scheduler_stops_after_ambiguous_submit(tmp_path,monkeypatch):
    c=controller();source=tmp_path/'source';write_json(source/'pilot/verification.json',{})
    calls=[]
    def run(args,**kw):calls.append(args);raise subprocess.TimeoutExpired(args,90)
    monkeypatch.setattr(c.subprocess,'run',run)
    with pytest.raises(RuntimeError,match='Check Slurm'):c.submit(source,tmp_path/'out','smoke')
    with pytest.raises(FileExistsError):c.submit(source,tmp_path/'out','smoke')
    assert len(calls)==1


def test_smoke_gate_prevents_switching_policy_or_changed_artifacts(tmp_path):
    c=controller();source=tmp_path/'source';write_json(source/'manifest.json',{})
    smoke=tmp_path/'smoke'
    write_json(smoke/'manifest.json',dict(protocol='fixed-policy-critics-v1',phase='smoke',
        source_manifest=sha256(source/'manifest.json'),runtime={}))
    write_json(smoke/'result.json',{})
    write_json(smoke/'verification.json',dict(passed=True,manifest_sha256=sha256(smoke/'manifest.json'),
                                           files={'result.json':sha256(smoke/'result.json')}))
    c.smoke_gate(smoke,source)
    write_json(source/'manifest.json',dict(new_actor=True))
    with pytest.raises(ValueError,match='Matching'):c.smoke_gate(smoke,source)
