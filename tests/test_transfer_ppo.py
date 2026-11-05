"""CPU protocol tests plus mocked pinned trainer observation-hook integration."""
import ast,importlib.util,json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1]


def module(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'tools/transfer'/f'{name}.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_two_commands_have_matched_policy_and_distinct_frozen_features(tmp_path):
    m=module('online_transfer');cfg=dict(model_path='/cache/model',filter_sha256='hash',layers=[27,26,24,20,12],rms_eps=1e-6)
    a=m.command(tmp_path,cfg,'slice',701,60,100,False,'slice')
    b=m.command(tmp_path,cfg,'rms_two_thirds',701,60,1000,False,'layer')
    for cmd in (a,b):
        assert 'experiment.learning_curve=true' in cmd and 'experiment.critic_folds=1' in cmd
        assert 'experiment.critic_statistics=cumulative' in cmd and 'trainer.total_training_steps=60' in cmd
        assert 'actor_rollout_ref.model.critic_feature_mode=frozen' in cmd
        assert 'trainer.test_freq=10' in cmd and 'experiment.evaluation_split=test' in cmd
    assert 'actor_rollout_ref.model.critic_feature_normalization=rms' in b
    assert 'actor_rollout_ref.model.critic_feature_layer=bottleneck' in a
    c=m.command(tmp_path,cfg,'slice',19,2,.01,True,'smoke')
    assert 'experiment.evaluation_split=val' in c and 'trainer.test_freq=1' in c


def test_observation_hook_restores_state_and_preserves_validation_metrics(tmp_path):
    # AST avoids importing unavailable local VERL; exercise the actual class against
    # a parent whose evaluate_test re-enters _validate exactly like MathTrainer.
    tree=ast.parse((ROOT/'src/math_rl/learning_curve.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
    events=[]
    class Parent:
        def _validate(self):events.append('validate');return {'normal_validation':True}
        def evaluate_test(self,label):
            events.append(label);self._validate()
            return dict(accuracy=.6,questions=2,truncation_rate=0.,seconds=.1)
        def fit(self):
            self.global_steps=10
            result=self._validate()
            assert result=={'normal_validation':True}
    (tmp_path/'linear-critic').mkdir()
    rows=[dict(step=i,transitions=100,fit_seconds=.01) for i in range(1,11)]
    (tmp_path/'linear-critic/metrics.jsonl').write_text('\n'.join(json.dumps(v) for v in rows))
    saved={}
    scope=dict(MathTrainer=Parent,json=json,time=SimpleNamespace(perf_counter=lambda:2.),write_json=lambda p,r:saved.update({p.name:r}))
    exec(compile(ast.Module(body=[cls],type_ignores=[]),'curve','exec'),scope)
    t=scope['LearningCurveTrainer']();t.output=tmp_path;t.completed_updates=0;t.total_training_steps=60
    t.config=SimpleNamespace(trainer=SimpleNamespace(test_freq=10),experiment=SimpleNamespace(evaluation_split='test'))
    t.fit()
    assert t.global_steps==10 and t.completed_updates==0 and not t._curve_active
    assert events==['validate','curve-10','validate']
    assert saved['curve-10.json']['response_tokens']==1000
    assert saved['curve-10.json']['answers']==640


def test_validation_selection_never_reads_test_scores(tmp_path):
    m=module('online_transfer')
    (tmp_path/'manifest.json').write_text('{}')
    def audit(case,steps,eps,arm,split):
        assert split=='val'
        return dict(initial_validation=dict(accuracy=.5),final_validation=dict(accuracy=.6),training_gpu_hours=.1)
    def read(path):
        assert 'test' not in path.name
        step=int(path.stem.split('-')[1]);return dict(accuracy=.5+step/100)
    saved={}
    with patch.object(m,'require_gate'),patch.object(m,'audit',audit),patch.object(m,'read',read),patch.object(m,'write_json',side_effect=lambda p,r:saved.update(r)):
        m.select(tmp_path,{})
    assert set(saved['epsilons'])==set(m.ARMS)
    assert set(saved['epsilons'].values())=={.01} # deterministic tie


def test_submission_smoke_gate_and_parallel_pilots(tmp_path):
    m=module('submit_online_transfer');calls=[]
    def run(cmd,**kwargs):calls.append(cmd);return SimpleNamespace(returncode=0,stdout=str(100+len(calls)),stderr='')
    with patch.object(m.subprocess,'run',run):m.submit(tmp_path/'out','smoke',tmp_path/'source')
    assert len(calls)==3 and '--gres=gpu:a5000:2' in calls[1]
    assert '--array=0-1%2' in calls[1] and '--dependency=afterok:102' in calls[2]
    assert not any('--gres' in s for s in calls[2])


def test_failed_curve_evaluation_restores_training_counters(tmp_path):
    tree=ast.parse((ROOT/'src/math_rl/learning_curve.py').read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
    class Parent:
        def _validate(self):return {}
        def evaluate_test(self,label):raise RuntimeError('evaluation failed')
    scope=dict(MathTrainer=Parent,json=json,time=SimpleNamespace(perf_counter=lambda:0.),write_json=lambda *a:None)
    exec(compile(ast.Module(body=[cls],type_ignores=[]),'curve','exec'),scope)
    t=scope['LearningCurveTrainer']();t._curve_active=True;t.global_steps=10;t.completed_updates=0;t.total_training_steps=60
    t.config=SimpleNamespace(trainer=SimpleNamespace(test_freq=10))
    with pytest.raises(RuntimeError,match='evaluation failed'):t._validate()
    assert t.global_steps==10 and t.completed_updates==0 and t._curve_active


@pytest.mark.parametrize('modern', [False, True])
def test_auc_works_with_legacy_and_modern_numpy(monkeypatch, modern):
    import numpy as np
    integrate = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    if modern:
        monkeypatch.setattr(np, 'trapezoid', integrate, raising=False)
        monkeypatch.delattr(np, 'trapz', raising=False)
    else:
        monkeypatch.setattr(np, 'trapz', integrate, raising=False)
        monkeypatch.delattr(np, 'trapezoid', raising=False)
    m = module('online_transfer')
    # Pilot scalar AUC and report per-question AUC use the same trapezoidal rule.
    assert m._trapezoid([0., .5, 1.], [0, 10, 20]) / 20 == pytest.approx(.5)
    curves = np.array([[0., 1.], [.5, .5], [1., 0.]])
    np.testing.assert_allclose(m._trapezoid(curves, [0, 10, 20], axis=0) / 20,
                               [.5, .5])
