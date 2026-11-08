import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest

ROOT=Path(__file__).resolve().parents[1]


def module():
    spec=importlib.util.spec_from_file_location('recovery',ROOT/'tools/diagnostics/recover_chunk_study.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_only_known_runtime_patch_is_allowed():
    m=module();old={p:'old' for p in m.ALLOWED_CHANGES};new=dict(old)
    new['packages/slice-critic/src/slice_critic/chunks.py']='new'
    assert m.changed_runtime(old,new)==['packages/slice-critic/src/slice_critic/chunks.py']
    new['configs/online.yaml']='new'
    with pytest.raises(ValueError,match='Unexpected runtime'):m.changed_runtime(old,new)


def test_submission_restarts_only_failed_tasks_after_four_smokes(tmp_path):
    m=module();source=tmp_path/'source';source.mkdir();out=tmp_path/'new'
    (source/'manifest.json').write_text(json.dumps(dict(arms={str(i):{} for i in range(4)},seeds=[901,902,903,904,905])))
    for i in range(20):
        (source/f'run-{i}-status.json').write_text(json.dumps(dict(state='failed' if i in (8,13) else 'complete')))
    calls=[]
    def run(cmd,**kw):
        calls.append(cmd);return SimpleNamespace(returncode=0,stdout=str(100+len(calls)),stderr='')
    with patch.object(m.subprocess,'run',run):
        m.submit(SimpleNamespace(source=source,out=out,concurrency=2))
    assert len(calls)==5
    assert '--array=0-3%2' in calls[1]
    assert '--array=8,13%2' in calls[3]
    assert '--dependency=afterok:103' in calls[3] and '--dependency=afterok:104' in calls[4]
    assert not out.exists()  # Only allocated prepare may create the study.
    with pytest.raises(FileExistsError):
        m.submit(SimpleNamespace(source=source,out=out,concurrency=2))


def test_prepare_preserves_original_and_hashes_links(tmp_path,monkeypatch):
    m=module()
    from math_rl import chunk_study as study
    from math_rl import provenance
    source=tmp_path/'old';source.mkdir();out=tmp_path/'new'
    for folder in ('data','smoke-data','filters'):(source/folder).mkdir()
    (source/'data/questions.json').write_text('[]');(source/'filters/sparse.pt').write_bytes(b'filter')
    key='packages/slice-critic/src/slice_critic/chunks.py'
    old=dict(runtime={key:'old'},data_hashes={'data/questions.json':m.digest(source/'data/questions.json')},
        filter_sha256=m.digest(source/'filters/sparse.pt'),steps=120,arms={'slice-chunk':dict(mode='math')})
    m.write(source/'manifest.json',old)
    monkeypatch.setattr(provenance,'snapshot',lambda root:dict(files={key:'new'}))
    monkeypatch.setattr(study,'jobs',lambda manifest:[('slice-chunk',901),('slice-chunk',903)])
    monkeypatch.setattr(study,'require_gate',lambda out:None)
    monkeypatch.setattr(study,'audit',lambda *a,**kw:dict(accounted_gpu_hours=1.))
    monkeypatch.setattr(study,'validate',lambda out:m.read(out/'manifest.json'))
    good=source/'slice-chunk-seed901';good.mkdir()
    names=['result.json','run.json','base-test.jsonl','final-test.jsonl',
        'linear-critic/metrics.jsonl','linear-critic/statistics.pt','linear-critic/120.pt','chunk-metrics.jsonl']
    for name in names:
        path=good/name;path.parent.mkdir(exist_ok=True,parents=True);path.write_text('{}')
    bad=source/'slice-chunk-seed903';bad.mkdir();(bad/'partial').write_text('preserve')
    (source/'slice-chunk-seed903.log').write_text('ValueError: Empty or unknown sampled ByteLevel token')
    m.write(source/'run-0-status.json',dict(state='complete'));m.write(source/'run-1-status.json',dict(state='failed'))
    original_manifest=(source/'manifest.json').read_bytes()
    m.prepare(source,out)
    assert (source/'manifest.json').read_bytes()==original_manifest
    assert (bad/'partial').read_text()=='preserve'
    assert (out/good.name).is_symlink() and not (out/bad.name).exists()
    assert m.validate(out)['recovery']['failed_tasks']==[1]
    (good/'result.json').write_text('changed')
    with pytest.raises(ValueError,match='Retained artifact changed'):m.validate(out)
