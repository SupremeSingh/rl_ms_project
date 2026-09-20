import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('migrate_critic_parser', ROOT / 'scripts/migrate_critic_parser.py')
migration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migration)


def fixture_run(tmp_path, monkeypatch):
    root, out = tmp_path / 'repo', tmp_path / 'run'
    (root / 'src/math_rl').mkdir(parents=True)
    (out / 'trajectories').mkdir(parents=True)
    (root / migration.GRADER).write_text('patched parser')
    (root / 'other.py').write_text('unchanged')
    monkeypatch.setattr(migration, 'ROOT', root)
    migration.write_json(out / 'questions.json', dict(questions=[dict(id='q0')]))
    migration.write_json(out / 'status.json', dict(stage='generate', state='failed'))
    migration.write_json(out / 'trajectories/00000.json', dict(question_id='q0', responses=[
        dict(token_ids=[4], score=1, verifier_status='correct')]))
    migration.write_json(out / 'manifest.json', dict(
        config=dict(protocol='frozen-critics-v2', responses=1), verifier=dict(packages={}),
        questions_sha256=migration.sha256(out / 'questions.json'),
        files={migration.GRADER: migration.OLD_GRADER_SHA256, 'other.py': migration.sha256(root / 'other.py')}))
    return root, out


def test_migration_preserves_data_and_records_exact_change(tmp_path, monkeypatch):
    _, out = fixture_run(tmp_path, monkeypatch)
    before = (out / 'manifest.json').read_bytes()
    data = (out / 'trajectories/00000.json').read_bytes()
    migration.migrate(out)
    assert (out / 'manifest.before-sympify-fix.json').read_bytes() == before
    assert (out / 'trajectories/00000.json').read_bytes() == data
    after = json.loads((out / 'manifest.json').read_text())
    old = json.loads(before)
    assert after['config'] == old['config']
    assert after['questions_sha256'] == old['questions_sha256']
    assert after['verifier']['parse_error_policy'] == migration.PARSE_ERROR_POLICY
    receipt = json.loads((out / 'parser-fix-migration.json').read_text())
    assert receipt['new_manifest_sha256'] == migration.sha256(out / 'manifest.json')
    assert receipt['preserved_trajectory_sha256']['00000.json'] == migration.sha256(out / 'trajectories/00000.json')
    migration.migrate(out)
    assert json.loads((out / 'manifest.json').read_text()) == after


@pytest.mark.parametrize('change', ['code', 'questions', 'active', 'unknown_grader', 'bad_score'])
def test_migration_rejects_unrelated_changes(tmp_path, monkeypatch, change):
    root, out = fixture_run(tmp_path, monkeypatch)
    if change == 'code':
        (root / 'other.py').write_text('different')
    elif change == 'questions':
        (out / 'questions.json').write_text('{}')
    elif change == 'active':
        migration.write_json(out / 'status.json', dict(stage='generate', state='running'))
    elif change == 'unknown_grader':
        manifest = json.loads((out / 'manifest.json').read_text())
        manifest['files'][migration.GRADER] = 'unknown'
        migration.write_json(out / 'manifest.json', manifest)
    else:
        migration.write_json(out / 'trajectories/00000.json', dict(question_id='q0', responses=[
            dict(token_ids=[4], score=1, verifier_status='parse_failure')]))
    before = (out / 'manifest.json').read_bytes()
    with pytest.raises(ValueError):
        migration.migrate(out)
    assert (out / 'manifest.json').read_bytes() == before
    assert not (out / 'manifest.before-sympify-fix.json').exists()
