import ast
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml

from math_rl import experiment as study
from math_rl.data import select_questions
from math_rl.provenance import sha256


def manifest():
    return dict(conditions=copy.deepcopy(study.METHODS), filters={'sparse': dict(path='/filter.pt',sha256='digest')},
        layers=[27,26,24,20,12],rms_eps=1e-6,critic_lambda=.9,steps=120,seeds=[301,302,303,304,305])


@pytest.mark.parametrize('name', study.METHODS)
def test_exact_final_command_contract_and_main_validation(tmp_path,name):
    m=manifest()
    cmd=study.command(tmp_path,m,name,301,120,profile='higher_lr')
    assert cmd[1:3] == ['-m','math_rl.main']
    assert 'trainer.total_training_steps=120' in cmd and 'experiment.replay_capacity=0' in cmd
    if m['conditions'][name]['method'] in ('lstd','ridge'):
        assert 'experiment.critic_statistics=cumulative' in cmd
        assert 'experiment.critic_lambda=0.9' in cmd
        assert 'actor_rollout_ref.model.critic_encoder_precision=fp32' in cmd
    if name=='ppo':
        assert 'experiment.ppo_profile=higher_lr' in cmd
        assert 'critic.optim.lr=5e-05' in cmd
        assert 'critic.ppo_epochs=2' in cmd
    # Execute the real validator without importing Ray/VERL on a CPU developer host.
    root=Path(__file__).resolve().parents[1]
    def merge(a,b):
        for k,v in b.items():
            if isinstance(v,dict): merge(a.setdefault(k,{}),v)
            else:a[k]=v
    config=yaml.safe_load((root/'configs/training.yaml').read_text())
    merge(config,yaml.safe_load((root/'configs/online.yaml').read_text()))
    config['critic'].update(strategy='fsdp',use_dynamic_bsz=False,ulysses_sequence_parallel_size=1)
    config['trainer'].update(critic_warmup=0,val_only=False)
    for arg in cmd[3:]:
        key,value=arg.split('=',1); current=config
        for part in key.split('.')[:-1]:current=current.setdefault(part,{})
        parsed=yaml.safe_load(value)
        if key.endswith('.lr'): parsed=float(value)
        current[key.split('.')[-1]]=parsed
    class Attr(dict):
        __getattr__=dict.__getitem__
    def node(x): return Attr({k:node(v) for k,v in x.items()}) if isinstance(x,dict) else x
    tree=ast.parse((root/'src/math_rl/main.py').read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='validate')
    scope=dict(distribution=lambda name: SimpleNamespace(read_text=lambda name:json.dumps(
        {'vcs_info':{'commit_id':'8d9e350ea58c7ad4b50dd14d9dcb50577242c55f'}})),
        VERL_COMMIT='8d9e350ea58c7ad4b50dd14d9dcb50577242c55f',json=json,sha256=lambda p:'digest')
    exec(compile(ast.Module(body=[function],type_ignores=[]),'validator','exec'),scope)
    scope['validate'](node(config))
    if name.startswith(('ridge','lstd')):
        config['experiment']['replay_capacity']=256
        with pytest.raises(ValueError):scope['validate'](node(config))


def test_smoke_and_task_gates_use_selected_profile_and_all_seeds(tmp_path,monkeypatch):
    m=manifest();(tmp_path/'manifest.json').write_text(json.dumps(m))
    (tmp_path/'selection.json').write_text(json.dumps(dict(profile='higher_lr')))
    calls=[]
    monkeypatch.setattr(study,'launch',lambda *a,**kw:calls.append((a,kw)))
    monkeypatch.setattr(study,'audit_health',lambda *a:dict(gates={
        k:True for k in ('complete','finite','nonzero_updates','zero_initial_values')}))
    study.smoke(tmp_path,m)
    assert len(calls)==6
    assert all(kw['profile']=='higher_lr' and kw['calibration'] for _,kw in calls)
    calls.clear()
    for task in range(30):study.run_task(tmp_path,m,task)
    assert {(a[2],a[3]) for a,kw in calls}=={(n,s) for n in study.METHODS for s in m['seeds']}
    gate=json.loads((tmp_path/'smoke.json').read_text());gate['ppo_profile']='more_fitting'
    (tmp_path/'smoke.json').write_text(json.dumps(gate))
    with pytest.raises(ValueError,match='Smoke gate failed'):study.run_task(tmp_path,m,0)


def test_changed_ppo_calibration_gets_its_own_smoke(tmp_path,monkeypatch):
    m=manifest();m['calibration_steps']=60
    (tmp_path/'smoke.json').write_text(json.dumps(dict(passed=True,ppo_profile='more_fitting')))
    calls=[]
    def launch(*args,**kwargs):
        calls.append((args,kwargs))
        return dict(initial_validation={'accuracy':.5},final_validation={'accuracy':.6},training_gpu_hours=1)
    monkeypatch.setattr(study,'launch',launch)
    monkeypatch.setattr(study,'audit_health',lambda *a:dict(pass_checks=True,
        gates={k:True for k in ('complete','finite','nonzero_updates','zero_initial_values')}))
    study.calibrate(tmp_path,m)
    # Declared tie break selects normalized; it gets a separate two-step check.
    assert calls[-1][1]['case']=='smoke-selected-normalized'
    assert json.loads((tmp_path/'smoke.json').read_text())['ppo_profile']=='normalized'


def test_full_paired_report_does_not_drop_failed_ppo_health(tmp_path,monkeypatch):
    m=manifest();m.update(steps=1,seeds=[301,302],scope='fixture',primary_comparisons=list(study.PRIMARY))
    (tmp_path/'data').mkdir()
    (tmp_path/'data/questions.json').write_text(json.dumps(dict(questions=[dict(id='q',ground_truth='1',split='test')])))
    (tmp_path/'selection.json').write_text(json.dumps(dict(profile='higher_lr',tuning_training_gpu_hours=1)))
    monkeypatch.setattr(study,'audit_health',lambda *a:dict(pass_checks=False))
    for name,setting in m['conditions'].items():
        for seed in m['seeds']:
            case=tmp_path/f'{name}-seed{seed}';case.mkdir()
            row=dict(id='q',ground_truth='1',score=1,verifier_status='correct',response_tokens=4,capped=False)
            for label in ('base','final'):(case/f'{label}-test.jsonl').write_text(json.dumps(row)+'\n')
            (case/'result.json').write_text(json.dumps(dict(updates=1,answers=64,initialization_seconds=1,
                initial_test=dict(accuracy=1,seconds=1),final_test=dict(accuracy=1,seconds=1),training_gpu_hours=1)))
            if setting['method'] in ('ridge','lstd'):
                (case/'linear-critic').mkdir()
                (case/'linear-critic/metrics.jsonl').write_text(json.dumps(dict(step=1,fit_seconds=1)))
    study.report(tmp_path,m)
    summary=json.loads((tmp_path/'summary.json').read_text())
    assert len(summary['methods'])==6 and all(summary['paired'][k]['mean_pp']==0 for k in study.PRIMARY)
    assert not summary['ppo_health']['301']['pass_checks']
    assert len((tmp_path/'audit.jsonl').read_text().splitlines())==12
