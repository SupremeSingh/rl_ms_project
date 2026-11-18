import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from math_rl import learning_study as study
ROOT=Path(__file__).resolve().parents[1]

def manifest():return dict(filter_sha256='fixed',layers=[27,26,24,20,12],rms_eps=1e-6,ppo_profile='higher_lr')

def test_curve_commands_preserve_algorithms_and_keep_every_actor():
    for arm in study.ARMS:
        cmd,case=study.command(Path('/new'),manifest(),arm,1201)
        assert 'math_rl.curve_entry' in cmd
        assert 'trainer.test_freq=10' in cmd and 'trainer.save_freq=10' in cmd
        assert 'trainer.max_actor_ckpt_to_keep=20' in cmd
        assert 'actor_rollout_ref.actor.checkpoint.save_contents=[hf_model]' in cmd
        assert 'trainer.val_before_train=false' in cmd
        assert 'experiment.evaluation_split=val' in cmd and 'algorithm.lam=0.95' in cmd
        assert 'trainer.total_training_steps=120' in cmd and 'data.seed=1201' in cmd
        assert not any('data/test.parquet' in x for x in cmd)
        if arm=='ppo':assert 'critic.optim.lr=5e-05' in cmd
        if arm=='grpo':assert 'algorithm.adv_estimator=grpo' in cmd

def test_actor_tasks_cover_all_fifteen_runs_once():
    tasks=study.actor_tasks()
    assert tasks==study.actor_tasks()
    assert len(tasks)==15
    assert {(t['seed'],t['arm']) for t in tasks}=={(s,a) for s in study.SEEDS for a in study.ARMS}
    assert [study.coordinates(i) for i in range(15)]==[(t['seed'],t['arm']) for t in tasks]
    assert [study.coordinates(i,True) for i in range(5)]==[(19,a) for a in study.ARMS]
    for invalid in (-1,15):
        with pytest.raises(ValueError):study.coordinates(invalid)

def test_curve_coverage_does_not_use_duplicate_final_evaluation(tmp_path):
    rows=[dict(step=i,label='base-validation' if i==0 else 'periodic',accuracy=.7,training_seconds=i*10) for i in range(0,121,10)]
    rows.append(dict(step=120,label='final-validation',accuracy=.7,training_seconds=0))
    (tmp_path/'learning-curve.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    assert len(study.curve_rows(tmp_path,120,10))==13
    rows.pop(3)
    (tmp_path/'learning-curve.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError,match='Missing'):study.curve_rows(tmp_path,120,10)

def test_driver_hooks_subtract_validation_and_checkpoint_time(tmp_path,monkeypatch):
    import sys
    from math_rl import curve_runtime
    clock=[0.]
    monkeypatch.setattr(curve_runtime.time,'perf_counter',lambda:clock[0])
    class Trainer:
        def __init__(self):self.output=tmp_path;self.global_steps=0
        def _validate(self):clock[0]+=2;return {'val-core/numina_numeric/reward/mean@1':.7}
        def evaluate_test(self,label):return self._validate()
        def _save_checkpoint(self):clock[0]+=3
        def fit(self):
            clock[0]+=10;self.global_steps=10;self._validate();self._save_checkpoint()
            clock[0]+=20;self.global_steps=20;self._validate()
    monkeypatch.setitem(sys.modules,'math_rl.trainer',SimpleNamespace(MathTrainer=Trainer))
    curve_runtime.install();t=Trainer();t.evaluate_test('base-validation');t.fit();t.evaluate_test('final-validation')
    rows=[json.loads(x) for x in (tmp_path/'learning-curve.jsonl').read_text().splitlines()]
    assert [r['label'] for r in rows]==['base-validation','periodic','periodic','final-validation']
    assert [r['training_seconds'] for r in rows]==[0,10,30,0]
    totals=json.loads((tmp_path/'training-clock.json').read_text())
    assert totals['training_seconds']==30 and totals['evaluation_seconds']==4 and totals['checkpoint_seconds']==3

def test_submission_has_fifteen_independent_actors_then_16_final_tests(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('submit_curves',ROOT/'scripts/submit_learning_study.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    out=tmp_path/'out';out.mkdir();(out/'manifest.json').write_text('{}')
    import hashlib
    (out/'smoke.json').write_text(json.dumps(dict(passed=True,manifest_sha256=hashlib.sha256(b'{}').hexdigest())))
    calls=[]
    monkeypatch.setattr(m.subprocess,'run',lambda args,**kw:(calls.append(args) or SimpleNamespace(returncode=0,stdout=str(len(calls)),stderr='')))
    m.submit(out,'main')
    assert len(calls)==4 and '--array=0-14%6' in calls[0]
    assert '--gres=gpu:a5000:2' in calls[0] and '--time=12:00:00' in calls[0]
    assert '--array=0-15%6' in calls[2]
    assert '--dependency=afterok:1' in calls[1]
    with pytest.raises(FileExistsError):m.submit(out,'main')


def test_submission_caps_concurrency_and_smoke_has_five_actors(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('submit_curves',ROOT/'scripts/submit_learning_study.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    calls=[]
    monkeypatch.setattr(m.subprocess,'run',lambda args,**kw:(calls.append(args) or SimpleNamespace(returncode=0,stdout=str(len(calls)),stderr='')))
    out=tmp_path/'smoke'
    with pytest.raises(ValueError):m.submit(out,'smoke',source=tmp_path,concurrency=7)
    assert not calls and not Path(str(out)+'-submission').exists()
    m.submit(out,'smoke',source=tmp_path,concurrency=2)
    assert len(calls)==3 and '--array=0-4%2' in calls[1]
    assert '--gres=gpu:a5000:2' in calls[1]
