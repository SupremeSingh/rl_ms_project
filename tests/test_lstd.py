"""Equation, terminal alignment and end-to-end cache tests for frozen LSTD."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from math_rl.lstd import accumulate, design, empty_statistics, solve
from math_rl.provenance import sha256, write_json


def test_equations_match_explicit_reference_and_chunking():
    mean, scale = torch.tensor([2., -1.]), torch.tensor([3., 2.])
    states = torch.tensor([[1., 2.], [3., 4.], [2., 8.], [999., -999.]])
    phi = design(states[:-1], mean, scale)
    following = torch.cat((phi[1:], torch.zeros(1, 3)), 0)
    for batch in (1, 2, 100):
        stats = empty_statistics(2)
        accumulate(stats, states, 1, mean, scale, batch)
        torch.testing.assert_close(stats['a'], phi.T @ (phi - following))
        torch.testing.assert_close(stats['b'], phi[-1])
        torch.testing.assert_close(stats['gram'], phi.T @ phi)
        torch.testing.assert_close(stats['returns_rhs'], phi.sum(0))
        assert stats['transitions'] == 3
        for method in ('lstd', 'ridge'):
            w, diagnostic = solve(stats, method, .01)
            a = stats['a' if method == 'lstd' else 'gram'] / 3
            b = stats['b' if method == 'lstd' else 'returns_rhs'] / 3
            penalty = torch.diag(torch.tensor([.01, .01, 0.], dtype=torch.float64))
            # Independent numpy reference; regularized solution need not equal true values.
            np.testing.assert_allclose(w.numpy(), np.linalg.solve((a + penalty).numpy(), b.numpy()), atol=1e-12)
            assert diagnostic['relative_residual'] < 1e-12


def test_exact_finite_chain_without_regularization():
    # Intercept plus one feature spans the two states exactly. Both eventually succeed.
    stats = empty_statistics(1)
    mean, scale = torch.zeros(1), torch.ones(1)
    accumulate(stats, torch.tensor([[0.], [1.], [999.]]), 1, mean, scale)
    weights, _ = solve(stats, 'lstd', 0.)
    torch.testing.assert_close(design(torch.tensor([[0.], [1.]]), mean, scale) @ weights,
                               torch.ones(2, dtype=torch.float64))


def test_terminal_and_episode_boundaries():
    stats = empty_statistics(1)
    mean, scale = torch.tensor([10.]), torch.tensor([2.])
    for x, reward in ((1., 1), (3., 0)):
        accumulate(stats, torch.tensor([[x], [12345.]]), reward, mean, scale)
    # All transitions terminal => LSTD and return regression equations identical.
    torch.testing.assert_close(stats['a'], stats['gram'])
    torch.testing.assert_close(stats['b'], stats['returns_rhs'])
    assert stats['trajectories'] == 2
    with pytest.raises(ValueError):
        accumulate(stats, torch.tensor([[float('nan')], [0.]]), 0, mean, scale)


def load_runner():
    path = Path(__file__).resolve().parents[1] / 'scripts/fit_lstd.py'
    spec = importlib.util.spec_from_file_location('fit_lstd', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_source(root):
    root.mkdir()
    (root / 'features').mkdir()
    (root / 'trajectories').mkdir()
    questions = [dict(id=f'q{i}', split=split) for i, split in enumerate(('train', 'val', 'test'))]
    write_json(root / 'questions.json', dict(questions=questions))
    write_json(root / 'manifest.json', dict(config=dict(protocol='frozen-critics-v2', exclusion_policy='test'),
                                          questions_sha256=sha256(root / 'questions.json')))
    write_json(root / 'summary.json', dict(scope='fixture'))
    torch.save(dict(mean=torch.zeros(2), scale=torch.ones(2), baseline=.5), root / 'normalization.pt')
    for i, q in enumerate(questions):
        rows = [dict(score=r, token_ids=[1, 2], verifier_status='correct' if r else 'incorrect',
                     finish_reason='length' if r else 'stop') for r in (1, 0)]
        rows.append(dict(score=None, token_ids=[3], verifier_status='parse_failure', finish_reason='stop'))
        raw = root / 'trajectories' / f'{i:05d}.json'
        write_json(raw, dict(question_id=q['id'], responses=rows))
        states = torch.tensor([[1., 0.], [1., 1.], [9., 9.], [-1., 0.], [-1., 1.], [9., 9.]])
        torch.save(dict(features=states, offsets=torch.tensor([0, 3, 6]), rewards=torch.tensor([1., 0.]),
                   response_indices=torch.tensor([0, 1]), exclusion_policy='test', question_id=q['id'],
                   trajectories_sha256=sha256(raw)), root / 'features' / f'{i:05d}.pt')
        if q['split'] != 'train':
            torch.save(dict(x=states[[0, 1, 3, 4]], y=torch.tensor([1., 1., 0., 0.]),
                       question=torch.full((4,), i), position=torch.tensor([0, 1, 0, 1])), root / f"{q['split']}.pt")
    return questions


def test_pipeline_resume_and_read_only_source(tmp_path):
    runner = load_runner()
    source, out = tmp_path / 'source', tmp_path / 'out'
    fixture_source(source)
    before = {str(p): sha256(p) for p in source.rglob('*') if p.is_file()}
    runner.run(source, out, [.001, .01])
    first = json.loads((out / 'summary.json').read_text())
    assert first['training_transitions'] == 4
    assert first['retained_training_answers'] == 2
    assert first['retained_capped_answers'] == 1
    assert first['training_verifier_status_counts']['parse_failure'] == 1
    for method in ('lstd', 'ridge'):
        assert first['methods'][method]['test']['accuracy'] == 1
        checkpoint = torch.load(out / f'{method}.pt', weights_only=True)
        assert checkpoint['state']['weight'].shape == (1, 2)
    runner.run(source, out, [.001, .01])
    assert json.loads((out / 'summary.json').read_text())['training_transitions'] == 4
    assert before == {str(p): sha256(p) for p in source.rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match='Resume'):
        runner.run(source, out, [.1])
    shard_path = source / 'features/00000.pt'
    shard = torch.load(shard_path, weights_only=True)
    shard['features'][0, 0] += 1
    torch.save(shard, shard_path)
    with pytest.raises(ValueError, match='changed'):
        runner.run(source, out, [.001, .01])


def test_bad_alignment_rejected(tmp_path):
    runner = load_runner()
    questions = fixture_source(tmp_path / 'source')
    path = tmp_path / 'source/features/00000.pt'
    shard = torch.load(path, weights_only=True)
    shard['offsets'][1] = 2
    torch.save(shard, path)
    with pytest.raises(ValueError, match='alignment'):
        runner.read_shard(tmp_path / 'source', 0, questions[0])


@pytest.mark.parametrize('trace_lambda', [0., .9, .99, .999, 1.])
def test_traces_against_sequential_reference(trace_lambda):
    generator = torch.Generator().manual_seed(1729)
    mean, scale = torch.tensor([1., 2., -3.]), torch.tensor([2., 4., 1.])
    episodes = [(torch.randn(18, 3, generator=generator).double(), 1),
                (torch.randn(8, 3, generator=generator).double(), 0)]
    a, b = torch.zeros(4, 4, dtype=torch.float64), torch.zeros(4, dtype=torch.float64)
    for states, outcome in episodes:
        phi = design(states[:-1], mean, scale)
        z = torch.zeros(4, dtype=torch.float64)
        for t in range(len(phi)):
            z = trace_lambda * z + phi[t]
            next_phi = phi[t + 1] if t + 1 < len(phi) else torch.zeros_like(z)
            a += torch.outer(z, phi[t] - next_phi)
            if t + 1 == len(phi):
                b += z * outcome
    for batch_size in (1, 3, 4096):
        stats = empty_statistics(3)
        for states, outcome in episodes:
            accumulate(stats, states, outcome, mean, scale, batch_size, trace_lambda)
        torch.testing.assert_close(stats['a'], a, atol=1e-11, rtol=1e-11)
        torch.testing.assert_close(stats['b'], b, atol=1e-11, rtol=1e-11)
        if trace_lambda == 1:
            torch.testing.assert_close(stats['a'], stats['gram'], atol=1e-11, rtol=1e-11)
            torch.testing.assert_close(stats['b'], stats['returns_rhs'], atol=1e-11, rtol=1e-11)
            for alpha in (0., 1e-6, .1):
                left, _ = solve(stats, 'lstd', alpha)
                right, _ = solve(stats, 'ridge', alpha)
                torch.testing.assert_close(left, right, atol=1e-10, rtol=1e-10)


def test_long_trace_does_not_overflow_or_leak_between_episodes():
    generator = torch.Generator().manual_seed(42)
    stats = empty_statistics(4)
    mean, scale = torch.zeros(4), torch.ones(4)
    for length, reward in ((2048, 1), (1, 0), (777, 0)):
        states = torch.randn(length + 1, 4, generator=generator).double()
        accumulate(stats, states, reward, mean, scale, batch_size=127, trace_lambda=1.)
    torch.testing.assert_close(stats['a'], stats['gram'], atol=1e-9, rtol=1e-11)
    torch.testing.assert_close(stats['b'], stats['returns_rhs'], atol=1e-10, rtol=1e-11)


def test_lambda_sweep_endpoint_and_resume(tmp_path):
    runner = load_runner()
    source, out = tmp_path / 'source', tmp_path / 'out'
    fixture_source(source)
    lambdas = [0., .9, .99, .999, 1.]
    runner.run(source, out, [0., .001, .01], lambdas)
    summary = json.loads((out / 'summary.json').read_text())
    assert len(summary['methods']) == 6
    assert summary['lambda_one_ridge_check']['validation_prediction_max_difference'] < 1e-6
    candidates = {k: v for k, v in summary['methods'].items() if k != 'ridge'}
    chosen = min(candidates, key=lambda k: candidates[k]['validation_brier'])
    assert summary['selected_lstd'] == chosen
    assert summary['methods'][chosen]['prediction_spread']['std'] >= 0
    saved = torch.load(out / 'selected-lstd.pt', weights_only=True)
    assert saved['method'] == chosen
    runner.run(source, out, [0., .001, .01], lambdas)
    assert json.loads((out / 'summary.json').read_text())['training_transitions'] == 4
    with pytest.raises(ValueError, match='Resume'):
        runner.run(source, out, [0., .001, .01], [0., 1.])


@pytest.mark.parametrize('values', [[], [-.1], [1.1], [float('nan')], [0., 0.]])
def test_invalid_lambda_grid(tmp_path, values):
    with pytest.raises(ValueError, match='lambda'):
        load_runner().run(tmp_path, tmp_path / 'out', [.1], values)
