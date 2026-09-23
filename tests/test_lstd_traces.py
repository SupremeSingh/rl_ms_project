import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import compare_lstd_traces as runner
from fit_lstd import read_shard, run as fit
from math_rl.critic_probe import PREFIX_SAMPLING, sample_shard
from math_rl.provenance import sha256, write_json
from test_lstd import fixture_source


def make_cache(path, source, original_fit, math=False):
    path.mkdir()
    (path / 'features').mkdir()
    (path / 'trajectories').mkdir()
    q = dict(id='math/test' if math else 'q2', split='test', source_index=0 if math else 2)
    if math:
        q['level'] = 5
    raw = json.loads((source / 'trajectories/00002.json').read_text())
    raw['question_id'] = q['id']
    write_json(path / 'trajectories/00000.json', raw)
    shard = torch.load(source / 'features/00002.pt', weights_only=True)
    shard.update(question_id=q['id'], trajectories_sha256=sha256(path / 'trajectories/00000.json'))
    torch.save(shard, path / 'features/00000.pt')
    data = sample_shard(shard, q['source_index'])
    torch.save(data, path / 'test.pt')
    write_json(path / 'manifest.json', dict(protocol='lstd-math-hard-v1' if math else 'lstd-seed-validation-v1',
        source=str(source), fit=str(original_fit), questions=[q], config=dict(prefix_sampling=PREFIX_SAMPLING),
        files={str(original_fit / 'manifest.json'): sha256(original_fit / 'manifest.json')}))
    _, _, fingerprint = read_shard(path, 0, q)
    write_json(path / 'summary.json', dict(feature_hashes={'0': fingerprint}))


def setup(tmp_path):
    source, original = tmp_path / 'source', tmp_path / 'original'
    fixture_source(source)
    fit(source, original, [.001], [.999])
    gsm, math = tmp_path / 'gsm', tmp_path / 'math'
    make_cache(gsm, source, original)
    make_cache(math, source, original, math=True)
    return source, gsm, math


def test_three_traces_two_datasets_resume_and_read_only_inputs(tmp_path):
    source, gsm, math = setup(tmp_path)
    before = {str(p): sha256(p) for folder in (source, gsm, math) for p in folder.rglob('*') if p.is_file()}
    out = tmp_path / 'comparison'
    runner.run(source, gsm, math, out)
    summary = json.loads((out / 'summary.json').read_text())
    assert set(summary['datasets']) == {'gsm8k_new_answers', 'math_transfer'}
    for result in summary['datasets'].values():
        assert set(result['methods']) == set(runner.METHODS)
        assert result['answers']['retained_answers'] == 2
        for name in runner.METHODS[:-1]:
            assert result['methods'][name]['versus_ridge']['questions'] == 1
    assert summary['datasets']['math_transfer']['methods']['lstd-0.85']['levels']['5']['questions'] == 1
    assert json.loads((out / 'fit/summary.json').read_text())['training_transitions'] == 4
    runner.run(source, gsm, math, out)
    assert before == {str(p): sha256(p) for folder in (source, gsm, math) for p in folder.rglob('*') if p.is_file()}
    assert json.loads((out / 'status.json').read_text())['state'] == 'complete'


def test_modified_test_probe_rejected_before_fitting(tmp_path):
    source, gsm, math = setup(tmp_path)
    path = gsm / 'test.pt'
    data = torch.load(path, weights_only=True)
    data['x'][0, 0] += 10
    torch.save(data, path)
    with pytest.raises(ValueError, match='probes differ'):
        runner.run(source, gsm, math, tmp_path / 'out')
    assert not (tmp_path / 'out/fit').exists()


def test_training_overlap_rejected(tmp_path):
    source, gsm, math = setup(tmp_path)
    path = gsm / 'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['questions'][0]['id'] = 'q0'
    write_json(path, manifest)
    with pytest.raises(ValueError, match='overlap'):
        runner.load_cache(gsm, source, 'lstd-seed-validation-v1')
