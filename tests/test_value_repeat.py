"""Multi-seed orchestration: original PPO training, paired questions and coverage."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import subprocess

import numpy as np
import pytest

from math_rl import value_repeat as repeat
from math_rl import value_audit as base
from math_rl.provenance import write_json

ROOT=Path(__file__).resolve().parents[1]


def manifest(seed):
    return dict(actor_seed=seed,steps=120,ppo_profile='higher_lr',layers=[27,26,24,20,12],
                filter_sha256='fixed',rms_eps=1e-6,budgets=repeat.BUDGETS,
                question_seed=48371,reference_seed=base.stable_seed(88213,'reference',seed))


def test_original_actor_training_commands_are_preserved(tmp_path):
    for seed in repeat.SEEDS:
        m=manifest(seed)
        for arm in base.ARMS:
            cmd,case=base.command(tmp_path,m,arm)
            assert f'data.seed={seed}' in cmd and 'trainer.total_training_steps=120' in cmd
            assert 'algorithm.lam=0.95' in cmd and 'experiment.evaluation_split=val' in cmd
            assert 'experiment.cumulative_epsilon=0.01' in cmd
            assert not any('fixed_policy' in x for x in cmd)
            if arm=='ppo':
                assert 'algorithm.adv_estimator=gae' in cmd
                assert 'critic.optim.lr=5e-05' in cmd
                assert 'critic.ppo_epochs=2' in cmd
                assert 'actor_rollout_ref.model.critic_feature_mode=actor' in cmd
            if arm=='grpo':assert 'algorithm.adv_estimator=grpo' in cmd


def test_questions_match_across_seeds_while_reference_streams_differ(tmp_path):
    write_json(tmp_path/'data/questions.json',dict(questions=[dict(id=str(i),split='test') for i in range(100)]))
    cohorts=[base.audit_questions(tmp_path,manifest(seed),'full') for seed in repeat.SEEDS]
    assert cohorts[0]==cohorts[1]==cohorts[2] and len(cohorts[0])==50
    assert len({manifest(seed)['reference_seed'] for seed in repeat.SEEDS})==3


def test_task_mapping_covers_18_actors_and_72_reference_tasks():
    train=[repeat.coordinates(i,'train','main') for i in range(18)]
    assert len(set(train))==18 and {v[0] for v in train}==set(repeat.SEEDS)
    refs=[repeat.coordinates(i,'mc','main') for i in range(72)]
    assert len(set(refs))==72 and {v[2] for v in refs}=={0,1,2,3}
    assert repeat.coordinates(24,'mc','main')==(1102,'slice-token',0)
    with pytest.raises(ValueError):repeat.coordinates(72,'mc','main')
    with pytest.raises(ValueError):repeat.coordinates(6,'train','smoke')


def summaries():
    result=[]
    for seed in repeat.SEEDS:
        panels={}
        for arm in base.ARMS:
            rows=[]
            for q in ('a','b'):
                for i in range(4):rows.append(dict(question_id=q,state_id=f'{q}/{i}',
                    response_position=i*10,prediction=.7,reference=.5))
            if seed==repeat.SEEDS[1]:rows=[r for r in rows if r['state_id']!='b/3']
            panels[arm]=rows
        result.append(dict(panels=panels))
    return result


def test_progress_uses_same_complete_case_questions_at_every_position():
    progress=repeat.ordinal_progress(summaries())
    assert progress['questions']==['a'] and progress['questions_per_position']==1
    for arm in base.ARMS:
        np.testing.assert_allclose(progress['values'][arm],np.full((3,4),.2))
    # Primary metrics retain b, even though the progress plot excludes it.
    ids,arrays=repeat.question_arrays(summaries())
    assert ids==['a','b'] and arrays['ppo'].shape==(3,2,2)


def test_bootstrap_pairs_seeds_and_questions_without_pseudoreplicating_states():
    _,arrays=repeat.question_arrays(summaries())
    r=repeat.paired(arrays,'slice-token','ppo')
    assert r['mean']==0 and r['interval']==[0.,0.] and len(r['differences_by_seed'])==3
    arrays['slice-token'][:,:,0]+=.1
    r=repeat.paired(arrays,'slice-token','ppo',.0125)
    np.testing.assert_allclose(r['interval'],[.1,.1])
    broken=summaries();broken[0]['panels']['ppo']=[]
    with pytest.raises(ValueError,match='pairing'):repeat.question_arrays(broken)


def controller():
    spec=importlib.util.spec_from_file_location('submit_value_repeat',ROOT/'scripts/submit_value_repeat.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_main_arrays_resource_sizes_and_dependencies(tmp_path,monkeypatch):
    c=controller();monkeypatch.setattr(c,'check_smoke',lambda *a:None);calls=[]
    def run(args,**kw):calls.append(args);return SimpleNamespace(returncode=0,stdout=str(len(calls)),stderr='')
    monkeypatch.setattr(c.subprocess,'run',run)
    c.submit(tmp_path/'out','main',concurrency=6)
    assert len(calls)==8
    assert '--array=0-17%6' in calls[0] and '--gres=gpu:a5000:2' in calls[0]
    assert '--mem=128G' in calls[0] and '--mem=64G' in calls[5]
    assert '--array=0-71%6' in calls[5] and '--gres=gpu:a5000:1' in calls[5]
    assert all('--partition=compsci-gpu' in row for row in calls)
    for i,row in enumerate(calls[1:],1):assert f'--dependency=afterok:{i}' in row


def test_submit_timeout_blocks_automatic_retries(tmp_path,monkeypatch):
    c=controller();source=tmp_path/'source';write_json(source/'manifest.json',{})
    calls=[]
    def run(args,**kw):calls.append(args);raise subprocess.TimeoutExpired(args,90)
    monkeypatch.setattr(c.subprocess,'run',run)
    with pytest.raises(RuntimeError,match='Check Slurm'):c.submit(tmp_path/'out','smoke',source)
    with pytest.raises(FileExistsError):c.submit(tmp_path/'out','smoke',source)
    assert len(calls)==1


def test_aggregate_keeps_seed_metrics_actor_results_and_matched_plot(tmp_path,monkeypatch):
    pytest.importorskip('matplotlib')
    write_json(tmp_path/'manifest.json',{})
    data=summaries()
    for seed,summary in zip(repeat.SEEDS,data):
        folder=repeat.child(tmp_path,seed);m=manifest(seed)
        write_json(folder/'manifest.json',m)
        write_json(folder/'full/summary.json',summary)
        write_json(folder/'full/verification.json',{})
        write_json(folder/'train-training.json',dict(runs={a:dict(training_gpu_hours=2.) for a in base.ARMS}))
        for arm in base.ARMS:
            case=base.checkpoint_paths(folder,m,arm,'full')[0]
            write_json(case/'result.json',dict(final_validation=dict(accuracy=.74),training_gpu_hours=2.))
    monkeypatch.setattr(base,'require_gate',lambda *a:None)
    repeat.aggregate(tmp_path,dict(scope='Synthetic CPU fixture.'),'main')
    summary=base.read(tmp_path/'main/summary.json')
    assert len(summary['metrics']['ppo']['mae_by_seed'])==3
    assert summary['metrics']['ppo']['actor_validation_accuracy_mean']==pytest.approx(.74)
    assert summary['primary']['slice-token_minus_ppo']['interval']==[0.,0.]
    assert summary['progress']['questions']==['a']
    assert (tmp_path/'main/value-scatter.pdf').read_bytes().startswith(b'%PDF')
    assert base.read(tmp_path/'main/verification.json')['passed']
