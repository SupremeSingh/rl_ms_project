import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
from math_rl import chunk_study as study

ROOT=Path(__file__).resolve().parents[1]


def controller():
    spec=importlib.util.spec_from_file_location('submit_chunk_comparison',ROOT/'scripts/submit_chunk_comparison.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def manifest():
    return dict(arms=study.ARMS,seeds=study.SEEDS,steps=120,filter_sha256='fixed',layers=[27,26,24,20,12],rms_eps=1e-6,
                chunk_fallback='fixed_tokens',chunk_fallback_tokens=32)


def test_matched_commands_change_only_representation_and_time_scale(tmp_path):
    m=manifest()
    for arm,setting in study.ARMS.items():
        cmd,case=study.command(tmp_path,m,arm,901)
        for expected in ['experiment.method=ridge','experiment.critic_folds=1','experiment.cumulative_epsilon=0.01',
            'algorithm.lam=0.95','trainer.total_training_steps=120','trainer.test_freq=30','experiment.evaluation_split=test']:
            assert expected in cmd
        for key in ('experiment.chunk_fallback_tokens','actor_rollout_ref.model.critic_chunk_fallback_tokens','critic.chunk_fallback_tokens'):
            assert f'{key}=32' in cmd
        assert f'experiment.chunking={setting["mode"]}' in cmd
        assert case==f'{arm}-seed901'
        smoke,_=study.command(tmp_path,m,arm,19,True)
        assert 'trainer.total_training_steps=2' in smoke
        assert f'experiment.test_file={tmp_path}/smoke-data/val.parquet' in smoke
        assert not any('smoke-data/test.parquet' in arg for arg in smoke)
    jobs=study.jobs(m)
    assert len(set(jobs))==20 and set(jobs)=={(a,s) for a in study.ARMS for s in study.SEEDS}
    assert [a for a,s in jobs if s==901]!=[a for a,s in jobs if s==902]


def test_four_smokes_have_afterok_gate_and_cpu_prepare(tmp_path):
    m=controller();calls=[]
    def run(args,**kw):calls.append(args);return SimpleNamespace(returncode=0,stdout=str(100+len(calls)),stderr='')
    with patch.object(m.subprocess,'run',run):m.submit(tmp_path/'out','smoke',tmp_path/'source',6,fallback_tokens=32)
    assert len(calls)==3 and '--array=0-3%4' in calls[1]
    assert '--dependency=afterok:101' in calls[1] and '--dependency=afterok:102' in calls[2]
    assert '--gres=gpu:a5000:2' in calls[1] and not any('--gres' in a for a in calls[0]+calls[2])
    assert calls[0][-1]=='32'


def test_fallback_changes_cannot_silently_override_locked_study(tmp_path):
    m=controller()
    with patch.object(m.subprocess,'run') as run:
        with pytest.raises(ValueError,match='sealed manifest'):
            m.submit(tmp_path/'out','train',fallback_tokens=32)
        with pytest.raises(ValueError,match='positive integer'):
            m.submit(tmp_path/'out','smoke',tmp_path/'source',fallback_tokens=0)
        run.assert_not_called()


def test_failed_submission_stops_and_retains_marker(tmp_path):
    m=controller();calls=[]
    def run(args,**kw):calls.append(args);return SimpleNamespace(returncode=1,stdout='',stderr='Socket timed out')
    with patch.object(m.subprocess,'run',run):
        with pytest.raises(RuntimeError,match='Check Slurm'):m.submit(tmp_path/'out','smoke',tmp_path/'source')
        with pytest.raises(FileExistsError):m.submit(tmp_path/'out','smoke',tmp_path/'source')
    assert len(calls)==1
    assert json.loads((tmp_path/'out-submission/prepare-submission.json').read_text())['state']=='ambiguous_or_failed'


def test_train_requires_gate_matching_manifest(tmp_path):
    m=controller();out=tmp_path/'out';out.mkdir()
    (out/'manifest.json').write_text('{}')
    (out/'smoke.json').write_text(json.dumps(dict(passed=True,manifest_sha256='wrong')))
    with pytest.raises(ValueError,match='Matching passed smoke'):m.submit(out,'train')
    assert not (tmp_path/'out-submission/.submission-train').exists()


def test_bootstrap_preserves_seed_question_pairing():
    d=np.full((5,12),.1)
    np.testing.assert_allclose(study.paired_interval(d),[10.,10.])
    assert study.paired_interval(np.zeros((5,12)),.0125)==[0.,0.]


def test_report_outputs_primary_intervals_costs_and_readable_csv(tmp_path,monkeypatch):
    m=manifest();m['primary']=['slice-chunk_minus_slice-token','layer-chunk_minus_layer-token']
    (tmp_path/'data').mkdir()
    (tmp_path/'data/questions.json').write_text(json.dumps(dict(questions=[dict(id=str(i),ground_truth='1',split='test') for i in range(4)])))
    monkeypatch.setattr(study,'require_gate',lambda *a:None)
    def audit(out,m,arm,seed,smoke=False):
        return dict(result=dict(seed=seed,final_test=dict(accuracy=.5),training_gpu_hours=2.),
            accounted_gpu_hours=2.2,segmentation_fallback_answers=0,response_tokens=1000,states=100 if arm.endswith('chunk') else 1000,fit_seconds=1.)
    monkeypatch.setattr(study,'audit',audit)
    def answers(path):
        return [dict(id=str(i),ground_truth='1',score=i%2,verifier_status='correct' if i%2 else 'incorrect',capped=False) for i in range(4)]
    monkeypatch.setattr(study.runner,'read_answers',answers)
    study.report(tmp_path,m)
    summary=json.loads((tmp_path/'summary.json').read_text())
    assert summary['primary']['slice']['adjusted_interval_pp']==[0.,0.]
    assert len((tmp_path/'results.csv').read_text().splitlines())==21
    assert 'not an isolated segmentation-only intervention' in (tmp_path/'report.txt').read_text()


def test_prepare_reuses_sealed_full_data_and_filter_without_old_actors(tmp_path,monkeypatch):
    import torch
    from math_rl.data import write_online_data
    from math_rl.provenance import sha256
    source=tmp_path/'source';source.mkdir();questions=[]
    def word(i):
        result=''
        for _ in range(4):result+=chr(97+i%26);i//=26
        return result
    for i,split in enumerate(['train']*2000+['val']*100+['test']*2000):
        questions.append(dict(id=f'q{i}',question=f'Calculate the quantity named {word(i)}.',
            ground_truth='1',split=split,source_index=i))
    write_online_data(source/'data',questions)
    (source/'filters').mkdir();filter=source/'filters/sparse.pt';torch.save({},filter)
    original=dict(data_hashes={p.name:sha256(p) for p in (source/'data').iterdir()},
        model_hashes={'config.json':'fixed'},layers=[27,26,24,20,12],rms_eps=1e-6,
        filters={'sparse':dict(path=str(filter),sha256=sha256(filter))})
    (source/'manifest.json').write_text(json.dumps(original))
    monkeypatch.setattr(study,'model_identity',lambda *a:{'config.json':'fixed'})
    seen=[];monkeypatch.setattr(study,'load_filter',lambda *a,**k:seen.append((a,k)))
    out=tmp_path/'new';study.prepare(source,out,fallback_tokens=32)
    locked=json.loads((out/'manifest.json').read_text())
    assert locked['counts']=={'train':2000,'val':100,'test':2000}
    assert locked['seeds']==study.SEEDS and len(seen)==1
    assert locked['chunk_fallback_tokens']==32 and locked['chunk_fallback']=='fixed_tokens'
    assert sha256(out/'filters/sparse.pt')==sha256(filter)
    assert not (out/'ridge-seed301').exists()
    assert study.validate(out)['steps']==120
    (out/'filters/sparse.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='Frozen model/filter changed'):study.validate(out)
