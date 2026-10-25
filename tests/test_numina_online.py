import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numina_online as study
from math_rl.bottleneck_features import bottleneck_prefixes, load_filter
from math_rl.sparse_bottleneck import Bottleneck, encode
from math_rl.critic_layers import sampled_block_states
from math_rl.critic_features import rms_coordinates
from math_rl.large_probe import family_key
from math_rl.provenance import sha256


class Block(nn.Module):
    def forward(self,x): return (x.cumsum(1)+.1,)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__();self.emb=nn.Embedding(20,3)
        self.layers=nn.ModuleList([Block() for _ in range(3)])
    def forward(self,input_ids,**kwargs):
        x=self.emb(input_ids)
        for block in self.layers:x=block(x)[0]
        return SimpleNamespace(last_hidden_state=x)


def test_online_filter_matches_offline_and_is_causal(tmp_path):
    torch.manual_seed(7)
    encoder=Encoder().eval().requires_grad_(False);layers=[2,1,0]
    model=Bottleneck(9,768);mean=torch.randn(9);scale=torch.ones(9)*2
    path=tmp_path/'filter.pt'
    torch.save(dict(state=model.state_dict(),mean=mean,scale=scale,width=768),path)
    fitted=load_filter(path,sha256(path))
    assert all(not p.requires_grad for p in fitted[0].parameters())
    tokens=torch.tensor([[1,2,3,4,5,6]])
    padded=torch.tensor([[0,1,2,3,4,5,6,0]]);mask=torch.tensor([[0,1,1,1,1,1,1,0]])
    blocks=sampled_block_states(encoder,tokens,3,3,[0,1,2],layers)
    x=torch.cat([rms_coordinates(blocks[l],1e-6) for l in layers],1)
    expected=encode(fitted[0],x,mean,scale,'cpu')
    actual=bottleneck_prefixes(encoder,padded,mask,4,fitted,layers,1e-6)[0]
    torch.testing.assert_close(torch.from_numpy(actual),expected)
    padded[0,6]=12  # final token is future relative to every captured state
    again=bottleneck_prefixes(encoder,padded,mask,4,fitted,layers,1e-6)[0]
    np.testing.assert_array_equal(actual,again)
    assert all(not b._forward_hooks for b in encoder.layers)
    with pytest.raises(ValueError,match='checksum'):load_filter(path,'changed')


def test_nine_commands_keep_actor_budget_and_cross_fitted_cumulative_features(tmp_path):
    m=dict(conditions=study.conditions(),layers=[27,26,24,20,12],rms_eps=1e-6,
        filters={kind:dict(path='/tmp/'+kind+'.pt',sha256='abc') for kind in ('plain','sparse')})
    assert len(m['conditions'])==9
    for name in m['conditions']:
        cmd=study.command(tmp_path,m,name,201,60,profile='higher_lr')
        assert 'trainer.total_training_steps=60' in cmd and 'data.seed=201' in cmd
        assert 'experiment.replay_capacity=0' in cmd and 'algorithm.adv_estimator=gae' in cmd
        if name=='ppo':assert 'experiment.ppo_profile=higher_lr' in cmd
        else:
            assert 'experiment.critic_statistics=cumulative' in cmd
            assert 'actor_rollout_ref.model.critic_feature_mode=frozen' in cmd
            assert 'actor_rollout_ref.model.critic_encoder_precision=fp32' in cmd
            assert 'experiment.critic_lambda=0.99' in cmd
        calibration=study.command(tmp_path,m,name,19,2,calibration=True)
        assert f'experiment.test_file={tmp_path}/data/val.parquet' in calibration


def test_split_excludes_all_prior_families_without_relaxing(monkeypatch):
    monkeypatch.setattr(study,'candidate',lambda row:(dict(question=row['q'],ground_truth='1'),None))
    class Tok:
        def encode(self,text,add_special_tokens=False):return [1,2]
    rows=[dict(q=q) for q in ['old question 1','old question 2','apples','oranges','pears','pears','plums']]
    questions,audit=study.select_questions(rows,{family_key('old question 999')},Tok(),dict(train=2,val=1,test=1))
    assert len({q['family'] for q in questions})==4
    assert not any(q['question'].startswith('old') for q in questions)
    assert audit['exclusions']['overlap_or_duplicate']==3
    with pytest.raises(ValueError,match='No fallback'):
        study.select_questions(rows,set(),Tok(),dict(train=10,val=1,test=1))


def test_filter_selection_uses_validation_only(tmp_path):
    source=tmp_path/'source';sparse=tmp_path/'sparse';out=tmp_path/'out'
    for d in (source,sparse,out):d.mkdir()
    (source/'questions.json').write_text('[]')
    (source/'manifest.json').write_text(json.dumps(dict(questions_sha256=sha256(source/'questions.json'))))
    (sparse/'protocol.json').write_text(json.dumps(dict(width=768,source_manifest=sha256(source/'manifest.json'))))
    selected={}
    for kind in ('plain','sparse'):
        for seed,score in [(42,.2),(43,.1)]:
            path=sparse/f'{kind}-{seed}.pt';torch.save(dict(kind=kind,width=768,seconds=1),path)
            selected[f'{kind}-{seed}']=dict(path=path.name,sha256=sha256(path),validation=dict(brier=score))
    (sparse/'screen.json').write_text(json.dumps(dict(selected=selected)))
    _,filters=study.lock_filters(source,sparse,out)
    assert all(v['selected'].endswith('43') for v in filters.values())
    # A failed preparation can reuse identical copies, but never changed weights.
    _,again=study.lock_filters(source,sparse,out)
    assert again == filters
    (out/'filters/plain.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='Existing filter differs'):
        study.lock_filters(source,sparse,out)
    # No test artifacts exist: selection must not depend on them.


@pytest.mark.parametrize("focused", [False, True, "comparison"])
def test_complete_report_has_paired_primary_comparisons_and_all_answers(tmp_path,monkeypatch,focused):
    m=dict(conditions=study.conditions(),seeds=[201,202,203,204,205],steps=1,scope='synthetic',
        primary_comparisons=['lstd-plain_minus_lstd-two_thirds','lstd-sparse_minus_lstd-two_thirds',
                             'ridge-plain_minus_ridge-two_thirds','ridge-sparse_minus_ridge-two_thirds'])
    if focused == 'comparison':
        available=study.conditions();available['grpo']=dict(method='grpo',representation='none')
        m.update(conditions={k:available[k] for k in study.COMPARISON_METHODS},
                 seeds=[301,302,303,304,305],primary_comparisons=list(study.COMPARISON_PRIMARY),critic_lambda=.9)
    elif focused:
        m.update(conditions={k: study.conditions()[k] for k in study.FOCUSED_METHODS},
                 seeds=[201,202,203], primary_comparisons=list(study.FOCUSED_COMPARISONS))
    (tmp_path/'data').mkdir()
    (tmp_path/'data/questions.json').write_text(json.dumps(dict(questions=[
        dict(id='a',ground_truth='1',split='test'),dict(id='b',ground_truth='2',split='test')])))
    (tmp_path/'selection.json').write_text(json.dumps(dict(profile='more_fitting',tuning_training_gpu_hours=1)))
    monkeypatch.setattr(study,'audit_health',lambda *args:dict(pass_checks=True))
    for name,setting in m['conditions'].items():
        for seed in m['seeds']:
            case=tmp_path/f'{name}-seed{seed}';case.mkdir()
            rows=[dict(id=q,ground_truth=str(i+1),score=i,verifier_status='correct' if i else 'incorrect',
                       response_tokens=4,capped=False) for i,q in enumerate(('a','b'))]
            for label in ('base','final'):
                (case/f'{label}-test.jsonl').write_text('\n'.join(map(json.dumps,rows)))
            (case/'result.json').write_text(json.dumps(dict(updates=1,answers=64,initialization_seconds=1,
                initial_test=dict(accuracy=.5,seconds=1),final_test=dict(accuracy=.5,seconds=1),training_gpu_hours=1)))
            if setting['method'] in ('ridge','lstd'):
                (case/'linear-critic').mkdir()
                (case/'linear-critic/metrics.jsonl').write_text(json.dumps(dict(step=1,fit_seconds=1)))
    study.report(tmp_path,m)
    summary=json.loads((tmp_path/'summary.json').read_text())
    assert len(summary['methods'])==len(m['conditions'])
    assert all(summary['paired'][key]['mean_pp']==0 for key in m['primary_comparisons'])
    assert len((tmp_path/'audit.jsonl').read_text().splitlines())==2*len(m['conditions'])*len(m['seeds'])
    assert 'PPO health passed for all seeds: True' in (tmp_path/'report.txt').read_text()


def test_large_budget_increases_questions_and_training_without_changing_arms():
    standard=study.study_budget('standard');large=study.study_budget('large')
    assert large['counts']==dict(train=3000,val=500,test=2000)
    assert all(large['counts'][s]>standard['counts'][s] for s in standard['counts'])
    assert large['steps']*64==12800 and large['calibration_steps']==60
    m=dict(conditions=study.conditions())
    cmd=study.command(Path('/tmp/large'),m,'ppo',201,large['steps'])
    assert 'trainer.total_training_steps=200' in cmd
    with pytest.raises(ValueError): study.study_budget('invalid')


def test_filter_cost_ignores_downstream_fit_files(tmp_path):
    trials=[]
    for kind in ('plain','sparse'):
        for lr in (.001,.0001):
            trials.append(dict(kind=kind,seed=42,lr=lr))
            torch.save(dict(seconds=2.5),tmp_path/f'{kind}-42-{lr}.pt')
        # These real downstream artifacts have weights/checks, not training time.
        torch.save(dict(weights=torch.ones(2),checks={}),tmp_path/f'{kind}-42-ridge-fit.pt')
        torch.save(dict(weights=torch.ones(2),checks={}),tmp_path/f'{kind}-42-lstd-fit.pt')
    (tmp_path/'screen.json').write_text(json.dumps(dict(trials=trials)))
    recorded=study.filter_training_trials(tmp_path)
    assert len(recorded)==4
    assert sum(t['seconds'] for t in recorded)==10
    # Missing metadata in an actual training trial must still fail explicitly.
    torch.save({},tmp_path/'plain-42-0.001.pt')
    with pytest.raises(KeyError,match='seconds'):
        study.filter_training_trials(tmp_path)


@pytest.mark.parametrize('focused', [False, True, 'comparison'])
def test_short_preparation_and_task_schedule(tmp_path, monkeypatch, focused):
    source=tmp_path/'source'; out=tmp_path/'short'
    (source/'data').mkdir(parents=True); out.mkdir()
    (source/'data/questions.json').write_text('{}')
    checkpoint=source/'filter.pt'; checkpoint.write_bytes(b'filter')
    m=dict(conditions=study.conditions(),answers_per_update=64,
           data_hashes={'questions.json':study.sha256(source/'data/questions.json')},
           filters={'plain':dict(path=str(checkpoint),sha256=study.sha256(checkpoint))},
           model_hashes={},provenance={'files':{}},seeds=[201,202,203,204,205],scope='Original.')
    (source/'manifest.json').write_text(json.dumps(m))
    (source/'smoke.json').write_text(json.dumps(dict(passed=True,conditions=list(study.conditions()))))
    (source/'selection.json').write_text(json.dumps({'profile':'more_fitting'}))
    (source/'result.json').write_text('{}')
    monkeypatch.setattr(study,'snapshot',lambda root:{'files':{'new':'hash'}})
    study.prepare_short(source,out,focused=focused is True,comparison=focused=='comparison')
    saved=json.loads((out/'manifest.json').read_text())
    assert saved['steps']==(120 if focused=='comparison' else 60)
    assert saved['seeds']==([301,302,303,304,305] if focused=='comparison' else [201,202,203])
    assert saved['preset']=='short' and not (out/'result.json').exists()
    assert len(saved['conditions'])==(6 if focused=='comparison' else 5 if focused else 9)
    if focused is True: assert saved['primary_comparisons']==list(study.FOCUSED_COMPARISONS)
    if focused=='comparison':
        assert not (out/'smoke.json').exists()
        assert saved['critic_lambda']==.9
        saved.update(layers=[27,26,24,20,12],rms_eps=1e-6)
        saved['filters']['sparse']=saved['filters']['plain']
        for name in saved['conditions']:
            cmd=study.command(out,saved,name,301,120)
            if name in ('ppo','grpo'):
                assert 'experiment.critic_statistics=cumulative' not in cmd
            else: assert 'experiment.critic_lambda=0.9' in cmd
            if name=='grpo': assert 'algorithm.adv_estimator=grpo' in cmd
    if focused=='comparison':
        with pytest.raises(ValueError,match='Smoke gate missing'):
            study.run_task(out,saved,0)
        gate=dict(passed=True,conditions=list(saved['conditions']),manifest_sha256='wrong')
        (out/'smoke.json').write_text(json.dumps(gate))
        with pytest.raises(ValueError,match='Smoke gate failed'):
            study.run_task(out,saved,0)
        gate['manifest_sha256']=study.sha256(out/'manifest.json')
        (out/'smoke.json').write_text(json.dumps(gate))
    calls=[]
    monkeypatch.setattr(study,'launch',lambda *args: calls.append(args))
    for task in range(len(saved['seeds'])*len(saved['conditions'])): study.run_task(out,saved,task)
    assert {(c[2],c[3]) for c in calls}=={(name,seed) for name in saved['conditions'] for seed in saved['seeds']}
    assert all(c[4]==saved['steps'] for c in calls)
    with pytest.raises(ValueError,match='Invalid task'): study.run_task(out,saved,len(calls))
    with pytest.raises(ValueError,match='empty'): study.prepare_short(source,out,focused)


def test_smoke_uses_selected_ppo_profile_and_validation_only(tmp_path, monkeypatch):
    available=study.conditions();available['grpo']=dict(method='grpo',representation='none')
    m=dict(conditions={name:available[name] for name in study.COMPARISON_METHODS})
    (tmp_path/'manifest.json').write_text(json.dumps(m))
    (tmp_path/'selection.json').write_text(json.dumps(dict(profile='higher_lr')))
    calls=[]
    monkeypatch.setattr(study,'launch',lambda *args,**kw: calls.append((args,kw)))
    study.smoke(tmp_path,m)
    assert len(calls)==6
    assert all(a[3:5]==(19,2) and kw['calibration'] and kw['profile']=='higher_lr' for a,kw in calls)
    gate=json.loads((tmp_path/'smoke.json').read_text())
    assert gate['passed'] and gate['ppo_profile']=='higher_lr'
    assert gate['manifest_sha256']==study.sha256(tmp_path/'manifest.json')
