import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import smoke_slice_ppo as smoke
from slice_critic.data import family_key


def test_select_excludes_fitting_families_and_preserves_splits():
    rows = [dict(id=str(i), question=q, split=s) for i, (q, s) in enumerate([
        ('Buy 2 pears', 'train'), ('Buy 8 pears', 'train'), ('Add apples', 'train'),
        ('Count oranges', 'train'), ('Add apples', 'val'), ('Divide bananas', 'val'),
        ('Count melons', 'test')])]
    manifest = dict(representation_training_families=[family_key('Buy 3 pears')], selection_families=[])
    selected = smoke.select_questions(rows, manifest, train_count=2, val_count=1)
    assert [q['question'] for q in selected] == ['Add apples', 'Count oranges', 'Divide bananas']
    assert [q['split'] for q in selected] == ['train', 'train', 'val']
    with pytest.raises(ValueError, match='eligible train'):
        smoke.select_questions(rows, manifest, train_count=3, val_count=1)


@pytest.mark.parametrize('method', ['ridge', 'lstd'])
def test_command_locks_package_basis_and_online_algorithm(tmp_path, method):
    exported = dict(path='/tmp/filter.pt', sha256='digest', layers=[27, 26, 24, 20, 12], rms_eps=1e-6)
    cmd = smoke.command(tmp_path, method, exported)
    for setting in ['experiment.critic_statistics=cumulative', 'experiment.critic_lambda=0.9',
                    'experiment.cumulative_epsilon=0.01', 'experiment.replay_capacity=0',
                    'actor_rollout_ref.model.critic_feature_mode=frozen',
                    'actor_rollout_ref.model.critic_encoder_precision=fp32',
                    'actor_rollout_ref.model.critic_feature_layer=bottleneck',
                    'experiment.evaluation_split=val', 'trainer.total_training_steps=2']:
        assert setting in cmd
    assert f'experiment.test_file={tmp_path}/data/val.parquet' in cmd
    assert 'actor_rollout_ref.model.critic_bottleneck_layers=[27,26,24,20,12]' in cmd
    assert not any('artifact' in s or 'calibration.json' in s for s in cmd)


def write_run(case, method):
    folder = case / 'linear-critic'
    folder.mkdir(parents=True)
    (case / 'result.json').write_text(json.dumps(dict(updates=2, answers=128,
        evaluation_split='val', training_gpu_hours=.1)))
    stats = dict(step=2, epoch=0, method=method, epsilon=.01, trace_lambda=.9,
        stats=[dict(a=torch.zeros(769, 769), b=torch.zeros(769), answers=64) for _ in range(2)])
    torch.save(stats, folder / 'statistics.pt')
    for step in (1, 2):
        torch.save(dict(heads=[dict(weights=torch.zeros(769)) for _ in range(2)]), folder / f'{step}.pt')
    (folder / 'metrics.jsonl').write_text('\n'.join(json.dumps(dict(reset=step==1, feature_epoch=0))
                                                  for step in (1, 2)))
    return stats


@pytest.mark.parametrize('method', ['ridge', 'lstd'])
def test_verify_real_trainer_checkpoint_format_and_accumulation(tmp_path, method):
    # Existing trainer files use torch.save, not the package's sidecar format.
    stats = write_run(tmp_path, method)
    assert smoke.verify_run(tmp_path, method)['passed']
    stats['stats'][0]['answers'] = 0
    torch.save(stats, tmp_path / 'linear-critic' / 'statistics.pt')
    with pytest.raises(ValueError, match='retain both updates'):
        smoke.verify_run(tmp_path, method)


def test_verify_rejects_matrix_reset(tmp_path):
    write_run(tmp_path, 'lstd')
    (tmp_path / 'linear-critic' / 'metrics.jsonl').write_text('\n'.join(
        json.dumps(dict(reset=True, feature_epoch=0)) for _ in range(2)))
    with pytest.raises(ValueError, match='unexpectedly reset'):
        smoke.verify_run(tmp_path, 'lstd')
