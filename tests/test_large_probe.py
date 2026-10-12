"""Large study: disjointness, complete-readout controls, selection and job graph."""
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch

from math_rl.large_probe import (ARMS, HEADS, assign_splits, candidate, family_key,
    paired_errors, prefix_positions, streaming_normalization)


def test_reference_eligibility_does_not_use_model_outcome():
    row = dict(source='cn_k12', synthetic=False, question_type='math-word-problem',
        problem_is_valid='Yes', solution_is_valid='Yes', problem='How much is 2 plus 3?',
        solution=r'The answer is \boxed{5}.', answer='5')
    assert candidate(row)[0]['ground_truth'] == '5'
    assert candidate(dict(row, answer='6'))[0] is None
    assert candidate(dict(row, synthetic=True))[0] is None
    assert candidate(dict(row, problem='Find all roots.'))[0] is None
    assert candidate(dict(row, solution=r'\boxed{5} and \boxed{6}'))[0] is None


def test_family_split_and_prefix_sampling_are_deterministic():
    assert family_key('Alice has 12 apples.') == family_key('ALICE has 42 apples!')
    rows = [dict(id=str(i), family=f'family {i}') for i in range(20)]
    counts = dict(calibration=2, screen=2, train=8, val=3, test=3)
    a = assign_splits(rows, counts); b = assign_splits(list(reversed(rows)), counts)
    assert a == b and len({q['family'] for q in a}) == len(a)
    with pytest.raises(ValueError, match='eligible families'):
        assign_splits(rows[:2], counts)
    for length in (1, 2, 100):
        p = prefix_positions(length, 19, 3)
        assert p == prefix_positions(length, 19, 3)
        assert len(p) == 8 and p[0] == 0 and all(0 <= x < length for x in p)


def test_streaming_normalization_matches_full_isotropic():
    from math_rl.direct_jlens import isotropic_normalization
    x = torch.randn(37, 8)
    actual = streaming_normalization(x.numpy(), batch_size=7)
    expected = isotropic_normalization(x)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)
    assert len(actual[1].unique()) == 1


def test_question_bootstrap_does_not_count_prefixes_as_independent():
    y = np.array([0., 1.]); a = np.array([[.1, .9]]); b = np.array([[.3, .7]])
    x = paired_errors(y, a, b, np.array([1, 2]), repetitions=100)
    repeated = paired_errors(np.repeat(y, 8), np.repeat(a, 8, axis=1),
        np.repeat(b, 8, axis=1), np.repeat([1, 2], 8), repetitions=100)
    assert x['questions'] == repeated['questions'] == 2
    assert x['mean'] == pytest.approx(repeated['mean'])
    assert x['mean'] < 0


def module(monkeypatch):
    scripts = Path(__file__).resolve().parents[1]/'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('large_study', scripts/'large_jlens_study.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_full_budget_and_test_selection_lock(tmp_path, monkeypatch):
    m = module(monkeypatch)
    import compare_jspace_probes as original
    monkeypatch.setattr(original, 'installed_lens', lambda: {})
    m.write(tmp_path/'lens-metadata.json', dict(environment={}))
    cfg = m.config(False)
    assert sum(cfg['counts'][s] for s in ('train', 'val', 'test'))*cfg['responses'] == 104000
    called = []
    monkeypatch.setattr(m, 'dataset', lambda *args: called.append(args))
    with pytest.raises(FileNotFoundError):
        m.report(tmp_path, dict(config=cfg), [])
    assert not called  # Never open test data with a missing validation selection.


def test_packing_uses_all_layers_and_full_scores(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from math_rl.direct_jlens import readout_factor, compact_readout
    from math_rl.provenance import sha256
    import math_rl.jspace as js
    m = module(monkeypatch)
    import compare_jspace_probes as original
    monkeypatch.setattr(original, 'installed_lens', lambda: {'fixture': True})
    torch.manual_seed(13)
    U = torch.randn(11, 4); gain = torch.rand(4)+.5
    monkeypatch.setattr(js, 'load_readout_weights', lambda _: {'lm_head.weight': U, 'model.norm.weight': gain})
    transport = lambda h, l: h*1.3
    monkeypatch.setitem(sys.modules, 'jlens', SimpleNamespace(JacobianLens=SimpleNamespace(load=lambda _: SimpleNamespace(transport=transport))))
    (tmp_path/'lens.pt').write_bytes(b'fixture')
    factor = readout_factor(U, gain)
    torch.save(factor, tmp_path/'readout-factor.pt')
    m.write(tmp_path/'lens-metadata.json', dict(environment={'fixture': True}, sha256=sha256(tmp_path/'lens.pt'),
                                              factor_sha256=sha256(tmp_path/'readout-factor.pt')))
    questions = []; hashes = {}; raw = {}
    (tmp_path/'features').mkdir()
    for i, split in enumerate(('train', 'val', 'test')):
        q = dict(index=i, split=split); questions.append(q)
        blocks = {l: torch.randn(8, 4) for l in (4, 3, 1)}; raw[split] = blocks
        torch.save(dict(blocks=blocks, y=torch.arange(8).float()%2, position=torch.arange(8),
                        response=torch.arange(8)), m.feature(tmp_path, q))
        hashes[m.feature(tmp_path, q).name] = sha256(m.feature(tmp_path, q))
    m.write(tmp_path/'extraction-0.json', dict(environment={'fixture': True}, features=hashes))
    manifest = dict(config={'shards': 1}, layers=[4, 3], two_thirds=1, dimension=4, rms_eps=1e-6)
    for index, arm in enumerate(ARMS):
        m.pack(tmp_path, manifest, questions, index)
        # All-token accumulation must use exactly the same feature map as evaluation.
        transformed = m.linear_features(arm, raw['test'], manifest,
            SimpleNamespace(transport=transport), factor['factor'].float(), gain)
        np.testing.assert_allclose(transformed.numpy(),
            np.load(tmp_path/'packed'/arm/'test-x.npy'), rtol=1e-5, atol=1e-5)
    x = np.load(tmp_path/'packed'/ARMS[0]/'test-x.npy')
    expected = torch.cat([compact_readout(raw['test'][4], factor['factor'].float(), 1e-6),
                         compact_readout(transport(raw['test'][3], 3), factor['factor'].float(), 1e-6)], 1)
    np.testing.assert_allclose(x, expected.numpy(), rtol=1e-5, atol=1e-5)
    for arm in ARMS:
        data = m.dataset(tmp_path, arm, 'test')
        torch.testing.assert_close(data['y'], torch.arange(8).float()%2)
    assert len(HEADS) == 3


def test_html_escapes_generated_code(tmp_path, monkeypatch):
    m = module(monkeypatch)
    path = tmp_path/'review.html'
    m.review_html(path, [dict(id='q', question='<script>bad</script>', reference='3', reference_solution='3',
        answer=dict(response='<img src=x>', verifier_status='correct', finish_reason='stop'))])
    assert '<script>' not in path.read_text() and '&lt;script&gt;' in path.read_text()


def test_shell_scripts_parse():
    root = Path(__file__).resolve().parents[1]
    for name in ('submit_large_jlens.sh', 'run_large_jlens_stage.sh'):
        subprocess.run(['bash', '-n', str(root/'scripts'/name)], check=True)


def test_native_context_budget(monkeypatch):
    m = module(monkeypatch)
    native = dict(max_position_embeddings=4096)
    cfg = m.config(False)
    assert cfg['max_tokens'] == 3584
    assert m.context_budget(cfg, native, [dict(prompt_token_ids=[1]*512)]) == 4096
    assert m.context_budget(m.config(True), native) == 1024
    with pytest.raises(ValueError, match='exceeds native context'):
        m.context_budget(dict(cfg, max_tokens=4096), native)
    with pytest.raises(ValueError, match='Saved prompt'):
        m.context_budget(cfg, native, [dict(prompt_token_ids=[1]*513)])


@pytest.mark.parametrize('blocker', [None, 'pending', 'code', 'questions'])
def test_context_repair(tmp_path, monkeypatch, blocker):
    from math_rl.provenance import sha256
    m = module(monkeypatch)
    monkeypatch.setattr(m, 'ROOT', tmp_path)
    model = tmp_path/'model'; model.mkdir()
    monkeypatch.setattr(m, 'MODEL', model)
    m.write(model/'config.json', dict(max_position_embeddings=4096))
    script = tmp_path/'scripts/large_jlens_study.py'; script.parent.mkdir(); script.write_text('fixed')
    current = {'scripts/large_jlens_study.py': sha256(script)}
    monkeypatch.setattr(m, 'snapshot', lambda _: dict(files=current))
    out = tmp_path/'study'; out.mkdir()
    m.write(out/'questions.json', [dict(prompt_token_ids=[1, 2])])
    old = dict(config=dict(m.config(False), version='full-readout-large-v2', max_tokens=4096),
        code={'scripts/large_jlens_study.py': 'previous-code-hash'},
        model_hashes={'config.json': sha256(model/'config.json')}, questions_sha256=sha256(out/'questions.json'))
    m.write(out/'manifest.json', old)
    original = (out/'manifest.json').read_bytes()
    if blocker == 'pending':
        m.write(out/'pending/00000.json', dict(response='already generated'))
    elif blocker == 'code':
        current['src/unrelated.py'] = 'unexpected'
    elif blocker == 'questions':
        m.write(out/'questions.json', [])
    if blocker:
        with pytest.raises(ValueError):
            m.repair_context(out)
        assert (out/'manifest.json').read_bytes() == original
        return
    m.repair_context(out)
    repaired = m.read(out/'manifest.json')
    assert repaired['config']['max_tokens'] == 3584
    assert repaired['questions_sha256'] == old['questions_sha256']
    assert (out/'manifest.before-context-fix.json').read_bytes() == original
    m.repair_context(out)


@pytest.mark.parametrize('blocker', [None, 'scored', 'code'])
def test_verifier_repair_preserves_pending_answers(tmp_path, monkeypatch, blocker):
    m = module(monkeypatch)
    out = tmp_path/'study'; model = tmp_path/'model'
    monkeypatch.setattr(m, 'MODEL', model)
    m.write(model/'config.json', dict(max_position_embeddings=4096))
    m.write(out/'questions.json', [])
    current = {'scripts/large_jlens_study.py': 'new', 'src/math_rl/verifier_batch.py': 'new'}
    monkeypatch.setattr(m, 'snapshot', lambda _: dict(files=current))
    m.write(out/'manifest.json', dict(config=m.config(False),
        questions_sha256=m.sha256(out/'questions.json'),
        model_hashes={'config.json': m.sha256(model/'config.json')},
        code={name: 'old' for name in current}))
    m.write(out/'pending/00000.json', dict(responses=[dict(response='4', token_ids=[4])]))
    original = (out/'manifest.json').read_bytes()
    pending = (out/'pending/00000.json').read_bytes()
    if blocker == 'scored':
        m.write(out/'trajectories/00000.json', {})
    if blocker == 'code':
        current['unrelated.py'] = 'new'
    if blocker:
        with pytest.raises(ValueError):
            m.repair_verifier(out)
        assert (out/'manifest.json').read_bytes() == original
    else:
        m.repair_verifier(out)
        assert (out/'manifest.before-verifier-fix.json').read_bytes() == original
        assert m.read(out/'manifest.json')['config'] == m.config(False)
        m.repair_verifier(out)
    assert (out/'pending/00000.json').read_bytes() == pending


def test_submission_graph_without_slurm(tmp_path):
    """Use a fake sbatch executable: no cluster work is submitted by this test."""
    import os
    import shutil
    root = Path(__file__).resolve().parents[1]
    (tmp_path/'scripts').mkdir(); (tmp_path/'bin').mkdir()
    shutil.copyfile(root/'scripts/submit_large_jlens.sh', tmp_path/'scripts/submit_large_jlens.sh')
    (tmp_path/'scripts/large_jlens_study.py').touch()
    for env in ('.venv', '.venv-jlens', '.venv-ppo-verifier'):
        p = tmp_path/env/'bin/python'; p.parent.mkdir(parents=True); p.touch(); p.chmod(0o755)
    fake = tmp_path/'bin/sbatch'
    fake.write_text('#!/usr/bin/env python3\nimport json, pathlib, sys\n'
        'p=pathlib.Path("calls.jsonl")\n'
        'n=len(p.read_text().splitlines()) if p.exists() else 0\n'
        'with p.open("a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\n'
        'print(100+n)\n')
    fake.chmod(0o755)
    env = dict(os.environ, PATH=str(tmp_path/'bin')+os.pathsep+os.environ['PATH'])
    subprocess.run(['bash', 'scripts/submit_large_jlens.sh', '--full', '--out', 'study'],
                   cwd=tmp_path, env=env, check=True, capture_output=True)
    calls = [json.loads(line) for line in (tmp_path/'calls.jsonl').read_text().splitlines()]
    assert len(calls) == 10
    assert '--dependency=afterok:101' in calls[2]  # Lens after screen.
    assert '--dependency=afterok:101' in calls[3]  # Generation independently after screen.
    assert '--dependency=afterok:103:102' in calls[4]  # Extraction waits for both.
    assert '--array=0-14%3' in calls[6]
    assert '--dependency=afterok:105' in calls[7]
    assert '--dependency=afterok:106:108' in calls[9]
    subprocess.run(['bash', 'scripts/submit_large_jlens.sh', '--full', '--out', 'study', '--resume-from', 'fit'],
                   cwd=tmp_path, env=env, check=True, capture_output=True)
    calls = [json.loads(line) for line in (tmp_path/'calls.jsonl').read_text().splitlines()]
    assert len(calls) == 14 and '--dependency=afterok:110:112' in calls[13]


def test_cpu_fit_and_report_end_to_end(tmp_path, monkeypatch):
    """Real BCE training on tiny matched arrays; substitute CPU for cluster CUDA."""
    from math_rl.provenance import sha256
    import math_rl.critic_probe as probe
    m = module(monkeypatch)
    import compare_jspace_probes as original
    monkeypatch.setattr(original, 'installed_lens', lambda: {})
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, *a, **kw: self)
    monkeypatch.setattr(torch.nn.Module, 'cuda', lambda self, *a, **kw: self)
    original_fit, original_predict = probe.fit_trial, probe.predict
    def cpu_predict(head, kind, data, mean, scale, device, **kw):
        return original_predict(head, kind, data, mean, scale, torch.device('cpu'), **kw)
    def cpu_fit(kind, train, val, mean, scale, device, *a, **kw):
        return original_fit(kind, train, val, mean, scale, torch.device('cpu'), *a, **kw)
    monkeypatch.setattr(probe, 'fit_trial', cpu_fit)
    monkeypatch.setattr(probe, 'predict', cpu_predict)
    torch.set_num_threads(1)
    m.write(tmp_path/'lens-metadata.json', dict(environment={}))
    cfg = m.config(True); cfg['width'] = 8; cfg['shards'] = 1
    manifest = dict(config=cfg, dimension=2, layers=[4, 3])
    questions = []
    for index, split in enumerate(('train', 'val', 'test')):
        q = dict(index=index, split=split, id=str(index), question='1+1?', ground_truth='2', reference_solution='2')
        questions.append(q)
        m.write(m.trajectory(tmp_path, q), dict(responses=[dict(verifier_status='correct', score=1.,
            token_ids=[1, 2], response='2', finish_reason='stop')]))
    for arm in ARMS:
        dest = tmp_path/'packed'/arm; dest.mkdir(parents=True)
        files = {}
        for index, split in enumerate(('train', 'val', 'test')):
            x = np.random.default_rng(index).normal(size=(16, 4 if arm.endswith('concat') else 2)).astype('float32')
            np.save(dest/f'{split}-x.npy', x)
            torch.save(dict(y=torch.arange(16).float()%2, position=torch.arange(16), response=torch.arange(16),
                            question=torch.arange(4).repeat_interleave(4)), dest/f'{split}-labels.pt')
            for suffix in ('x.npy', 'labels.pt'):
                path = dest/f'{split}-{suffix}'; files[path.name] = sha256(path)
        m.write(dest/'complete.json', dict(files=files))
    real_dataset = m.dataset
    def no_test_before_lock(out, arm, split):
        if split == 'test':
            assert (tmp_path/'selection-lock.json').exists()
        return real_dataset(out, arm, split)
    monkeypatch.setattr(m, 'dataset', no_test_before_lock)
    from math_rl.large_linear import statistics, accumulate
    stats, normalization = {}, {}
    for arm in ARMS:
        train = m.dataset(tmp_path, arm, 'train')
        mean, scale = streaming_normalization(train['x'].numpy())
        normalization[arm] = (mean, scale)
        stats[arm] = statistics(train['x'].shape[1])
        for i in range(0, len(train['y']), 2):
            accumulate(stats[arm], train['x'][i:i+2], (i//2)%2, mean, scale)
    (tmp_path/'linear-statistics').mkdir()
    checkpoint = tmp_path/'linear-statistics/shard-0.pt'
    torch.save(dict(stats=stats, normalization=normalization), checkpoint)
    m.write(checkpoint.with_suffix('.json'), dict(sha256=sha256(checkpoint)))
    for task in range(len(ARMS)):
        m.linear_fit(tmp_path, manifest, task)
    for task in range(len(ARMS)*len(HEADS)):
        m.fit(tmp_path, manifest, task)
    m.report(tmp_path, manifest, questions)
    summary = json.loads((tmp_path/'summary.json').read_text())
    assert len(summary['results']) == 25 and len(summary['comparisons']) == 25
    assert all(np.isfinite(v['means']['brier']) for v in summary['results'].values())
    assert (tmp_path/'responses.html').exists()
