import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
analysis = importlib.import_module('analyze_lstd')
validation = importlib.import_module('validate_lstd_seed')
from math_rl.critic_probe import sample_shard
from math_rl.provenance import sha256, write_json


def fixture_data(out):
    (out / 'trajectories').mkdir(parents=True)
    (out / 'features').mkdir()
    q = dict(id='test-question', split='test', source_index=4500)
    raw_path = out / 'trajectories/00000.json'
    responses = [dict(token_ids=list(range(length)), score=reward,
                      verifier_status='correct' if reward else 'incorrect', finish_reason=reason)
                 for length, reward, reason in [(1, 1, 'stop'), (513, 0, 'length')]]
    responses.append(dict(token_ids=[1], score=None, verifier_status='parse_failure', finish_reason='stop'))
    write_json(raw_path, dict(question_id=q['id'], responses=responses))
    features = torch.zeros(516, 2)
    features[:2, 0] = 1
    features[2:, 0] = -1
    shard = dict(features=features, offsets=torch.tensor([0, 2, 516]), rewards=torch.tensor([1., 0.]),
                 response_indices=torch.tensor([0, 1]), exclusion_policy='test', question_id=q['id'],
                 trajectories_sha256=sha256(raw_path))
    torch.save(shard, out / 'features/00000.pt')
    return [q], sample_shard(shard, q['source_index'])


def test_metadata_alignment_and_remaining_never_enter_features(tmp_path):
    questions, data = fixture_data(tmp_path)
    before = data['x'].clone()
    meta, info = analysis.metadata(tmp_path, questions, data)
    assert info['total_answers'] == 3 and info['retained_answers'] == 2
    assert info['statuses']['parse_failure'] == 1
    np.testing.assert_array_equal(meta['remaining'], meta['length'] - meta['position'])
    assert meta['remaining'].min() >= 1
    assert np.all(meta['position'][:8] == 0)  # One-token response repeats question-only prefix.
    assert np.all(meta['capped'][8:] == 1)
    torch.testing.assert_close(data['x'], before)
    data['position'][0] = 123
    with pytest.raises(ValueError, match='position'):
        analysis.metadata(tmp_path, questions, data)


def test_fixed_bins_and_question_cluster_weighting():
    meta = dict(y=np.array([1., 1., 0.]), question=np.array([0, 0, 1]), answer=np.array([0, 0, 0]),
                length=np.array([128, 129, 513]), position=np.array([0, 128, 512]),
                remaining=np.array([128, 1, 1]), capped=np.array([0, 0, 1]))
    masks = analysis.groups(meta)
    assert masks['length/1_128'].tolist() == [True, False, False]
    assert masks['length/129_512'].tolist() == [False, True, False]
    assert masks['length/513_plus'].tolist() == [False, False, True]
    row = analysis.paired(meta, np.array([0., 0., 0.]), np.array([1., 1., 1.]), np.ones(3, bool))
    assert row['question_mean_difference'] == 0  # +1 for question 0, -1 for question 1.
    assert row['questions'] == 2 and row['sparse']
    assert analysis.paired(meta, np.zeros(3), np.zeros(3), np.zeros(3, bool))['prefixes'] == 0


def fixture_heads(root):
    fit, source = root / 'fit', root / 'source'
    fit.mkdir(); source.mkdir()
    norm = dict(mean=torch.zeros(2), scale=torch.ones(2), baseline=.5)
    torch.save(norm, source / 'normalization.pt')
    for name in ('selected-lstd.pt', 'ridge.pt'):
        torch.save(dict(weights=torch.tensor([.3, 0., .5], dtype=torch.float64),
                        alpha=.001, trace_lambda=.999 if name.startswith('selected') else None,
                        method='lstd-0.999', **norm), fit / name)
    write_json(fit / 'summary.json', dict(selected_lstd='lstd-0.999',
                                        paired_question_brier_selected_lstd_minus_ridge=-.00095))
    return fit, source


def test_locked_head_seed_evaluation_and_no_refitting(tmp_path):
    out = tmp_path / 'fresh'
    questions, _ = fixture_data(out)
    fit, source = fixture_heads(tmp_path)
    locked = {str(p): sha256(p) for p in fit.iterdir()}
    manifest = dict(fit=str(fit), source=str(source), selected_lstd='lstd-0.999',
                    files=locked, config=dict(generation_seed=314159))
    validation.evaluate(out, manifest, questions)
    report = json.loads((out / 'summary.json').read_text())
    assert report['paired']['question_mean_difference'] == 0
    assert report['answers']['retained_answers'] == 2
    assert report['questions'] == 1
    assert locked == {str(p): sha256(p) for p in fit.iterdir()}
    assert (out / 'length-report.txt').exists()
    (fit / 'ridge.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='critic changed'):
        validation.evaluate(out, manifest, questions)


def test_new_generation_seed_reaches_sampler(tmp_path, monkeypatch):
    import math_rl.ppo_reward as reward_module
    import frozen_critics
    seen = []
    class Engine:
        def __init__(self, **kwargs):
            pass
        def generate(self, prompts, params):
            seen.append(params['seed'])
            return [SimpleNamespace(outputs=[SimpleNamespace(text='1', token_ids=[1],
                                                              finish_reason='stop', stop_reason=None)])]
    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(LLM=Engine, SamplingParams=lambda **kw: kw))
    monkeypatch.setattr(reward_module, 'compute_score', lambda *a, **kw:
                        [dict(score=1, verifier_status='correct')])
    (tmp_path / 'trajectories').mkdir()
    questions = [dict(id=str(i), ground_truth='1', prompt_token_ids=[5]) for i in range(2)]
    frozen_critics.generate(tmp_path, dict(responses=1, generation_seed=314159), questions)
    assert seen == [314159, 314160]
    frozen_critics.generate(tmp_path, dict(responses=1, generation_seed=314159), questions)
    assert seen == [314159, 314160]  # Resume does not regenerate saved answers.


def test_seed_manifest_locks_inputs_and_rejects_original_seed(tmp_path, monkeypatch):
    fit, source = fixture_heads(tmp_path)
    from math_rl.critic_probe import PREFIX_SAMPLING
    original = dict(config=dict(protocol='frozen-critics-v2', prefix_sampling=PREFIX_SAMPLING,
                    sampling=dict(temperature=1., top_p=1., top_k=-1, max_tokens=2048,
                                  seed='42 + question index'), responses=16),
                    versions={p: 'pinned' for p in ('torch', 'transformers', 'vllm', 'datasets', 'numpy')},
                    verifier=dict(rule='locked'), files={})
    write_json(source / 'manifest.json', original)
    write_json(source / 'questions.json', dict(questions=[dict(id='q', split='test')]))
    hashes = {name: sha256(source / name) for name in ('manifest.json', 'questions.json', 'normalization.pt')}
    write_json(fit / 'manifest.json', dict(source=str(source), source_hashes=hashes))
    monkeypatch.setattr(validation, 'version', lambda _: 'pinned')
    monkeypatch.setattr(validation, 'verifier_command', lambda: ['verifier'])
    monkeypatch.setattr(validation.subprocess, 'check_output', lambda *a, **kw: '{"rule": "locked"}')
    out = tmp_path / 'replication'
    saved = validation.initialize(fit, out, 314159)
    assert saved['questions'][0]['source_index'] == 0
    assert saved['config']['responses'] == 16
    assert saved == validation.initialize(fit, out, 314159)
    with pytest.raises(ValueError, match='overlap'):
        validation.initialize(fit, tmp_path / 'bad-seed', 42)
    with pytest.raises(ValueError, match='Resume'):
        validation.initialize(fit, out, 314160)
    (fit / 'ridge.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Resume'):
        validation.initialize(fit, out, 314159)
