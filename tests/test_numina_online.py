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
from math_rl.direct_jlens import rms_coordinates
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
    # No test artifacts exist: selection must not depend on them.


def test_complete_report_has_paired_primary_comparisons_and_all_answers(tmp_path,monkeypatch):
    m=dict(conditions=study.conditions(),seeds=[201,202,203,204,205],steps=1,scope='synthetic',
        primary_comparisons=['lstd-plain_minus_lstd-two_thirds','lstd-sparse_minus_lstd-two_thirds',
                             'ridge-plain_minus_ridge-two_thirds','ridge-sparse_minus_ridge-two_thirds'])
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
            if setting['method']!='ppo':
                (case/'linear-critic').mkdir()
                (case/'linear-critic/metrics.jsonl').write_text(json.dumps(dict(step=1,fit_seconds=1)))
    study.report(tmp_path,m)
    summary=json.loads((tmp_path/'summary.json').read_text())
    assert len(summary['methods'])==9
    assert all(summary['paired'][key]['mean_pp']==0 for key in m['primary_comparisons'])
    assert len((tmp_path/'audit.jsonl').read_text().splitlines())==90
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
