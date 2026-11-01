import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH=Path(__file__).resolve().parents[1]/'tools/diagnostics/start_verified_critic_training.py'
spec=importlib.util.spec_from_file_location('verified_submit',PATH)
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)


def setup(tmp_path,monkeypatch):
    monkeypatch.setattr(helper,'ROOT',tmp_path)
    monkeypatch.setenv('USER','testuser')
    out=tmp_path/'study';(out/'.submission-chain/run').mkdir(parents=True)
    m=dict(protocol='slice-seven-critics-gpu-v1',steps=120,seeds=[501,502,503],
           conditions={str(i):{} for i in range(7)},provenance={'files':{}},tools={})
    (out/'manifest.json').write_text(json.dumps(m))
    (out/'smoke.json').write_text(json.dumps(dict(passed=True,checks=m['conditions'],
        manifest_sha256=helper.digest(out/'manifest.json'))))
    (out/'jobs.tsv').write_text('smoke\t101\ngate\t102\n')
    (out/'.submission-chain/run/submission.json').write_text(json.dumps(dict(state='submitting',
        command=['sbatch','--dependency=afterok:102'])))
    return out


def test_recovers_only_explicit_rejection_and_preserves_runtime(tmp_path,monkeypatch):
    out=setup(tmp_path,monkeypatch);calls=[]
    original=(out/'manifest.json').read_bytes()
    def run(cmd,**kw):
        calls.append(cmd)
        return SimpleNamespace(stdout='' if cmd[0]!='sbatch' else str(200+len(calls)))
    helper.start(out,recover=True,run=run)
    assert calls[0][0]=='squeue' and calls[1][0]=='sacct'
    assert not any(a.startswith('--dependency=') for a in calls[2])
    assert '--dependency=afterok:203' in calls[3]
    assert (out/'manifest.json').read_bytes()==original
    assert len(list((out/'.submission-chain').glob('run-rejected-*')))==1
    with pytest.raises(ValueError,match='already recorded'):helper.start(out,recover=True,run=run)


def test_no_force_retry_or_retry_if_accounting_finds_a_job(tmp_path,monkeypatch):
    out=setup(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='acknowledged'):helper.start(out)
    calls=[]
    def run(cmd,**kw):
        calls.append(cmd);return SimpleNamespace(stdout='123|RUNNING')
    with pytest.raises(ValueError,match='job exists'):helper.start(out,recover=True,run=run)
    assert len(calls)==1 and (out/'.submission-chain/run').exists()


def test_rejects_bad_gate_without_any_submission(tmp_path,monkeypatch):
    out=setup(tmp_path,monkeypatch)
    record=json.loads((out/'smoke.json').read_text());record['passed']=False
    (out/'smoke.json').write_text(json.dumps(record))
    with pytest.raises(ValueError,match='matching'):helper.start(out,recover=True)
