import json
from pathlib import Path
import sys

import pytest
import torch

from math_rl.ppo_baseline import PPO_PROFILES, zero_value_head, critic_diagnostics, audit_health, select_profile
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import math_comparison
import math_rigorous


def test_zero_head_keeps_pretrained_backbone_and_causal_values():
    from transformers import Qwen2Config, Qwen2ForTokenClassification
    config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, num_labels=1)
    model = Qwen2ForTokenClassification(config).eval()
    saved = {n: p.detach().clone() for n, p in model.named_parameters() if not n.startswith('score.')}
    zero_value_head(model)
    output = model(torch.tensor([[1, 2, 3]])).logits
    assert output.abs().sum() == 0
    for n, p in model.named_parameters():
        if n in saved:
            torch.testing.assert_close(saved[n], p)
    with pytest.raises(ValueError, match='exactly one'):
        zero_value_head(torch.nn.Linear(3, 1))


def test_diagnostics_ignore_padding_and_are_detached():
    before = torch.tensor([[0., 9.], [0., 0.]], requires_grad=True)
    after = torch.tensor([[.8, -9.], [.8, .8]], requires_grad=True)
    returns = torch.ones(2, 2)
    mask = torch.tensor([[1., 0.], [1., 1.]])
    d = critic_diagnostics(before, after, returns, mask)
    assert d['mse_before'] == 1
    assert d['mse_after'] == pytest.approx(.04)
    assert before.grad is None and after.grad is None


def test_health_requires_complete_learning_not_just_job_success(tmp_path):
    assert not audit_health(tmp_path, 2)['pass_checks']
    row = dict(mse_before=1., mse_after=.5, prediction_change=.1, grad_norm_max=.2,
               lr=1e-5, initial_value_abs_max=0., audit_seconds=2.)
    for rank in (0, 1):
        (tmp_path / f'rank-{rank}.jsonl').write_text('\n'.join(json.dumps(dict(row, update=i)) for i in (1, 2)))
    health = audit_health(tmp_path, 2)
    assert health['pass_checks']
    assert health['extra_forward_wall_seconds_estimate'] == 4.
    assert not audit_health(tmp_path, 3)['pass_checks']
    row['mse_after'] = 3.
    (tmp_path / 'rank-0.jsonl').write_text('\n'.join(json.dumps(dict(row, update=i)) for i in (1, 2)))
    assert not audit_health(tmp_path, 2)['pass_checks']


def test_selection_uses_validation_health_and_declared_tie_break():
    rows = [dict(profile=p, health=dict(pass_checks=True), initial_validation=dict(accuracy=.5),
                 final_validation=dict(accuracy=.6), final_test=dict(accuracy=1. if p == 'higher_lr' else 0.))
            for p in PPO_PROFILES]
    assert select_profile(rows) == 'normalized'
    rows[0]['health']['pass_checks'] = False
    assert select_profile(rows) == 'higher_lr'
    for r in rows:
        r['final_validation']['accuracy'] = .4
    with pytest.raises(RuntimeError, match='not launched'):
        select_profile(rows)


def test_calibration_command_never_evaluates_test_and_no_buffer(tmp_path):
    cmd = math_comparison.command(tmp_path, 'ppo', 17, 30, ppo_profile='normalized',
                                  evaluation_split='val', case_name='calibration-normalized')
    assert cmd[0] == sys.executable
    assert f'experiment.test_file={tmp_path}/data/val.parquet' in cmd
    assert not any('data/test.parquet' in s for s in cmd)
    assert 'experiment.replay_capacity=0' in cmd
    assert 'critic.loss_agg_mode=seq-mean-token-mean' in cmd
    for method in ('ridge', 'lstd', 'grpo'):
        cmd = math_comparison.command(tmp_path, method, 42, 60, ppo_profile='higher_lr')
        assert not any(s.startswith('critic.optim.lr=') for s in cmd)
        assert 'experiment.replay_capacity=0' in cmd


def test_locked_runner_stops_before_comparison_when_calibration_fails(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(math_comparison, 'prepare', lambda *a: dict(scope='test'))
    monkeypatch.setattr(math_rigorous, 'snapshot', lambda *a: {})
    calls = []
    def launch(*args):
        calls.append(args)
        return dict(initial_validation=dict(accuracy=.5), final_validation=dict(accuracy=.4))
    monkeypatch.setattr(math_rigorous, 'launch', launch)
    monkeypatch.setattr(math_rigorous, 'audit_health', lambda *a: dict(pass_checks=True))
    out = tmp_path / 'run'
    with pytest.raises(RuntimeError, match='not launched'):
        math_rigorous.run(SimpleNamespace(out=out, source=tmp_path, steps=60, calibration_steps=30, seeds=[42,43,44]))
    assert len(calls) == 3
    assert all(c[5] is True for c in calls)
    assert json.loads((out / 'status.json').read_text())['state'] == 'failed'


def test_pinned_normalized_loss_is_padding_and_microbatch_invariant():
    pytest.importorskip('verl')
    from verl.trainer.ppo.core_algos import compute_value_loss
    v = torch.tensor([[.2, .2, .2], [.2, .2, .2]], requires_grad=True)
    mask = torch.tensor([[1., 0., 0.], [1., 1., 1.]])
    def loss(x, m):
        return compute_value_loss(vpreds=x, values=torch.zeros_like(x), returns=torch.ones_like(x),
                                  response_mask=m, cliprange_value=.5, loss_agg_mode='seq-mean-token-mean')[0]
    whole = loss(v, mask)
    micro = sum(loss(v[i:i+1], mask[i:i+1]) for i in range(2))/2
    torch.testing.assert_close(whole, micro)
    torch.testing.assert_close(whole, loss(torch.nn.functional.pad(v,(0,4)), torch.nn.functional.pad(mask,(0,4))))
    whole.backward()
    assert torch.isfinite(v.grad).all() and v.grad[~mask.bool()].abs().sum() == 0


def test_runner_locks_profile_and_runs_every_method_seed(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(math_comparison, 'prepare', lambda *a: dict(scope='inspected'))
    monkeypatch.setattr(math_rigorous, 'snapshot', lambda *a: {})
    calls = []
    def launch(out, method, seed, steps, profile='legacy', calibration=False, name=None):
        calls.append((method, seed, steps, profile, calibration))
        if not calibration:
            assert (out / 'selection.json').is_file()
        return dict(initial_validation=dict(accuracy=.5), final_validation=dict(accuracy=.6), training_gpu_hours=1.)
    monkeypatch.setattr(math_rigorous, 'launch', launch)
    monkeypatch.setattr(math_rigorous, 'audit_health', lambda *a: dict(pass_checks=True))
    def report(out):
        assert len(calls) == 15
        (out / 'report.txt').write_text('comparison\n')
        return dict(complete=True, comparisons={'42': {'identical_base_scores': True}})
    monkeypatch.setattr(math_comparison, 'report', report)
    monkeypatch.setattr(math_rigorous, 'aggregate', lambda *a: {})
    out = tmp_path / 'run'
    math_rigorous.run(SimpleNamespace(out=out, source=tmp_path, steps=60, calibration_steps=30, seeds=[42,43,44]))
    assert calls[:3] == [('ppo', 17, 30, p, True) for p in PPO_PROFILES]
    assert {(m, seed) for m, seed, _, _, _ in calls[3:]} == {(m,s) for m in math_comparison.METHODS for s in [42,43,44]}
    assert all(p == ('normalized' if m == 'ppo' else 'legacy') for m, _, _, p, _ in calls[3:])
    assert json.loads((out / 'status.json').read_text())['state'] == 'complete'


def test_seed_summary_uses_paired_questions_and_seeds(tmp_path):
    for method in math_comparison.METHODS:
        for seed in [42,43,44]:
            folder = tmp_path / f'{method}-seed{seed}'
            folder.mkdir()
            scores = [1, 1 if method == 'lstd' else 0]
            rows = [dict(id=str(i), score=s, verifier_status='correct' if s else 'incorrect') for i,s in enumerate(scores)]
            (folder / 'final-test.jsonl').write_text('\n'.join(map(json.dumps, rows)))
            (folder / 'result.json').write_text(json.dumps(dict(training_gpu_hours=1.,
                initial_test=dict(accuracy=.5), final_test=dict(accuracy=sum(scores)/2))))
    (tmp_path / 'ppo-health.json').write_text(json.dumps({str(seed):
        dict(extra_forward_wall_seconds_estimate=180.) for seed in [42,43,44]}))
    summary = math_rigorous.aggregate(tmp_path, [42,43,44])
    assert summary['methods']['ppo']['diagnostic_gpu_hours_estimate_mean'] == pytest.approx(.1)
    assert summary['methods']['ppo']['gpu_hours_excluding_diagnostic_forward_estimate_mean'] == pytest.approx(.9)
    assert summary['methods']['lstd']['accuracy_mean'] == 1
    assert summary['methods']['lstd']['accuracy_seed_sd'] == 0
    assert summary['paired']['lstd_minus_ppo']['mean_pp'] == 50
    assert summary['paired']['lstd_minus_ppo']['seed_differences_pp'] == [50]*3


def test_pinned_baseline_worker_and_profile_compose():
    pytest.importorskip('verl')
    from hydra import compose, initialize_config_dir
    from math_rl.ppo_baseline_worker import AuditedCriticWorker
    from verl.workers.fsdp_workers import CriticWorker
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(config_dir=str(root / 'configs'), version_base=None):
        config = compose(config_name='math_online', overrides=['experiment.ppo_profile=normalized',
            'critic.loss_agg_mode=seq-mean-token-mean', 'critic.use_dynamic_bsz=false'])
    assert config.critic.ulysses_sequence_parallel_size == 1
    assert config.critic.strategy == 'fsdp'
    assert config.experiment.replay_capacity == 0
    assert issubclass(AuditedCriticWorker, CriticWorker)


def reuse_fixture(tmp_path):
    old, new = tmp_path / 'old', tmp_path / 'new'
    old.mkdir()
    (new / 'data').mkdir(parents=True)
    q = dict(id='math/val/0', ground_truth='2', split='val')
    (new / 'data/questions.json').write_text(json.dumps(dict(questions=[q])))
    manifest = dict(protocol='math-online-calibrated-v1',
        calibration=dict(seed=17, steps=30, candidates=PPO_PROFILES),
        data=dict(questions_sha256='questions', model_hashes={'weights': 'weights'}, data_hashes={'train.parquet': 'train', 'val.parquet': 'val', 'test.parquet': 'test'}),
        provenance=dict(files={'src/core.py': 'same', 'scripts/math_rigorous.py': 'new'}))
    (old / 'manifest.json').write_text(json.dumps(manifest))
    case = old / 'calibration-normalized'
    case.mkdir()
    result = dict(method='ppo', seed=17, updates=30, answers=1920, ppo_profile='normalized', evaluation_split='val')
    (case / 'result.json').write_text(json.dumps(result))
    (case / 'run.json').write_text(json.dumps(dict(provenance=dict(files={
        'src/core.py': 'same', 'scripts/math_rigorous.py': 'old'}),
        data_hashes={'train': 'train', 'val': 'val'},
        config=dict(experiment=dict(method='ppo', ppo_profile='normalized', evaluation_split='val', replay_capacity=0),
                    data=dict(seed=17), trainer=dict(total_training_steps=30),
                    critic=dict(optim=dict(lr=1e-5), ppo_epochs=2, loss_agg_mode='seq-mean-token-mean')))))
    for label in ('base-validation', 'final-validation'):
        (case / f'{label}-test.jsonl').write_text(json.dumps(dict(id=q['id'], ground_truth='2', score=1, verifier_status='correct')))
    (old / 'calibration-more_fitting').mkdir()
    (old / 'calibration-more_fitting/error.txt').write_text('socket too long')
    return old, new, manifest


def test_reuse_completed_candidates_preserves_failed_source(tmp_path):
    old, new, manifest = reuse_fixture(tmp_path)
    before = (old / 'calibration-normalized/result.json').read_bytes()
    reused = math_rigorous.reuse_calibration(old, new, manifest)
    assert set(reused) == {'normalized'}
    assert (new / 'calibration-normalized').is_symlink()
    assert not (new / 'calibration-more_fitting').exists()
    assert (old / 'calibration-more_fitting/error.txt').read_text() == 'socket too long'
    assert (old / 'calibration-normalized/result.json').read_bytes() == before


def test_reuse_rejects_training_changes(tmp_path):
    old, new, manifest = reuse_fixture(tmp_path)
    manifest['provenance']['files']['src/core.py'] = 'changed'
    with pytest.raises(ValueError, match='Training code'):
        math_rigorous.reuse_calibration(old, new, manifest)


def test_launch_uses_short_unique_ray_paths(tmp_path, monkeypatch):
    captured = []
    def run(cmd, **kwargs):
        captured.append(kwargs['env']['RAY_TMPDIR'])
        case = tmp_path / 'calibration-more_fitting'
        case.mkdir(exist_ok=True)
        (case / 'result.json').write_text(json.dumps(dict(updates=30, answers=1920)))
    monkeypatch.setattr(math_rigorous.subprocess, 'run', run)
    math_rigorous.launch(tmp_path, 'ppo', 17, 30, 'more_fitting', True, 'calibration-more_fitting')
    ray_path = captured[0]
    assert len((ray_path + '/ray/session_2026-09-27_16-39-31_806402_652187/sockets/plasma_store').encode()) <= 107
    assert 'more_fitting' not in ray_path
