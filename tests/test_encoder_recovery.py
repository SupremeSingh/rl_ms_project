import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools/encoder'))
import recover_compare as recovery


def seal(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    path.with_suffix(path.suffix+'.sha256').write_text(json.dumps({'sha256':recovery.digest(path)}))


def test_block_normalization_matches_dense_and_constant_input():
    torch.manual_seed(123)
    for x in (torch.randn(37,11), torch.ones(37,11), torch.randn(37,11)*.01+100):
        mean, scale = recovery.bounded_normalization(x, batch=7)
        expected_mean = x.mean(0)
        expected_scale = ((x-expected_mean).double().square().mean().sqrt().clamp_min(1e-8)).float()
        torch.testing.assert_close(mean, expected_mean)
        torch.testing.assert_close(scale, expected_scale.expand_as(mean), rtol=2e-4, atol=1e-7)
    with pytest.raises(ValueError):
        recovery.bounded_normalization(torch.empty(0,3))


def test_prepare_reuses_sealed_fits_preserves_partial_and_submits_nothing(tmp_path):
    source, out = tmp_path/'source', tmp_path/'recovery'
    m = dict(adapt_seeds=[42], compressor_seeds=[42], shards=1, runtime={'original':'hash'})
    seal(source/'manifest.json', m); seal(source/'questions.json', [])
    for b in ('causal', 'adapted-42'):
        seal(source/f'extract-{b}-0.json', {'passed':True})
    for name in ('answers','features'):
        (source/name).mkdir()
    fit = source/'fits/causal-A-0'
    fit.mkdir(parents=True); (fit/'model.pt').write_bytes(b'checkpoint')
    seal(fit/'complete.json', {'sha256':recovery.digest(fit/'model.pt')})
    partial = source/'fits/causal-C-42'
    partial.mkdir(); (partial/'unfinished').write_text('keep')
    before = {str(p.relative_to(source)):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    recovery.prepare(source, out)
    plan = json.loads((out/'recovery.json').read_text())
    assert plan['reused_fits'] == ['causal-A-0']
    assert plan['missing_fit_tasks'] == list(range(1,8))
    assert (out/'fits/causal-A-0').is_symlink()
    assert not (out/'fits/causal-C-42').exists()
    assert before == {str(p.relative_to(source)):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    assert not (out/'recovery-jobs.tsv').exists()
    with pytest.raises(FileExistsError):
        recovery.prepare(source, out)


def test_changed_completed_fit_is_rejected_before_creation(tmp_path):
    source, out = tmp_path/'source', tmp_path/'recovery'
    seal(source/'manifest.json', dict(adapt_seeds=[], compressor_seeds=[42], shards=1, runtime={}))
    seal(source/'questions.json', [])
    seal(source/'extract-causal-0.json', {'passed':True})
    fit = source/'fits/causal-A-0'
    seal(fit/'complete.json', {'sha256':'wrong'})
    (fit/'model.pt').write_bytes(b'bad')
    with pytest.raises(ValueError, match='Changed fit'):
        recovery.prepare(source, out)
    assert not out.exists()


def test_disk_cache_preserves_sampling_and_excludes_validation(tmp_path):
    from types import SimpleNamespace
    import numpy as np
    import shutil
    rows = [dict(key='a', split='train'), dict(key='b', split='train'), dict(key='v', split='val')]
    values = {k:dict(x=torch.arange(12*7680, dtype=torch.float32).reshape(12,7680),
                   y=torch.tensor([0.,1.]*6)) for k in ('a','b')}
    c = SimpleNamespace(LAYERS=list(range(5)), items=lambda *args:rows,
        feature=lambda out,b,row:values[row['key']], xvalue=lambda r,a:r['x'],
        stable_seed=lambda *args:123)
    x, y = recovery.cache_training(c, tmp_path, {}, 'causal', 'C', tmp_path)
    ids = torch.randperm(12, generator=torch.Generator().manual_seed(123))[:8]
    torch.testing.assert_close(x, torch.cat([values[k]['x'][ids] for k in ('a','b')]))
    torch.testing.assert_close(y, torch.cat([values[k]['y'][ids] for k in ('a','b')]))
    assert (tmp_path/'x.f32').exists()


def test_validation_report_uses_completed_fits_only(tmp_path):
    source, out = tmp_path/'source', tmp_path/'report'
    seal(source/'manifest.json', dict(adapt_seeds=[42]))
    for backbone, value in [('causal',.15), ('adapted-42',.17)]:
        folder = source/'fits'/f'{backbone}-A-0'
        folder.mkdir(parents=True)
        torch.save(dict(validation_brier=value), folder/'model.pt')
        seal(folder/'complete.json', dict(sha256=recovery.digest(folder/'model.pt')))
    recovery.validation_report(source, out)
    summary = json.loads((out/'summary.json').read_text())
    assert summary['validation_brier']['causal'] == {'A':.15}
    assert '+0.02000000' in (out/'report.txt').read_text()


def test_submission_defaults_to_dry_run(tmp_path, monkeypatch, capsys):
    import submit_recovery as submit
    seal(tmp_path/'manifest.json', dict(adapt_seeds=[], compressor_seeds=[42], shards=1))
    (tmp_path/'recovery.json').write_text('{}')
    monkeypatch.setattr(sys, 'argv', ['submit_recovery.py','--out',str(tmp_path)])
    def forbidden(*args, **kwargs):
        raise AssertionError('Dry run must not submit jobs')
    monkeypatch.setattr(submit.subprocess, 'run', forbidden)
    submit.main()
    text = capsys.readouterr().out
    assert 'Dry run only' in text and '--array=0,1,2,3%2' in text
    assert not (tmp_path/'recovery-jobs.tsv').exists()
