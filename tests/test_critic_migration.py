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
    for name in migration.PREVIOUS_FILE_HASHES:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text('patched code')
    (root / 'other.py').write_text('unchanged')
    monkeypatch.setattr(migration, 'ROOT', root)
    migration.write_json(out / 'questions.json', dict(questions=[dict(id='q0')]))
    migration.write_json(out / 'status.json', dict(stage='generate', state='failed'))
    migration.write_json(out / 'trajectories/00000.json', dict(question_id='q0', responses=[
        dict(token_ids=[4], score=1, verifier_status='correct')]))
    migration.write_json(out / 'manifest.json', dict(
        config=dict(protocol='frozen-critics-v2', responses=1), verifier=dict(packages={}),
        questions_sha256=migration.sha256(out / 'questions.json'),
        files={**{name: sorted(hashes)[0] for name, hashes in migration.PREVIOUS_FILE_HASHES.items()},
               'other.py': migration.sha256(root / 'other.py')}))
    return root, out


def test_migration_preserves_data_and_records_exact_change(tmp_path, monkeypatch):
    _, out = fixture_run(tmp_path, monkeypatch)
    before = (out / 'manifest.json').read_bytes()
    data = (out / 'trajectories/00000.json').read_bytes()
    migration.migrate(out)
    assert (out / 'manifest.before-exclusion-fix.json').read_bytes() == before
    assert (out / 'trajectories/00000.json').read_bytes() == data
    after = json.loads((out / 'manifest.json').read_text())
    old = json.loads(before)
    assert after['config'] == dict(old['config'], exclusion_policy=migration.EXCLUSION_POLICY)
    assert after['questions_sha256'] == old['questions_sha256']
    assert after['verifier']['parse_error_policy'] == migration.PARSE_ERROR_POLICY
    assert after['verifier']['timeout_policy'] == migration.TIMEOUT_POLICY
    receipt = json.loads((out / 'exclusion-fix-migration.json').read_text())
    assert receipt['new_manifest_sha256'] == migration.sha256(out / 'manifest.json')
    assert receipt['preserved_trajectory_sha256']['00000.json'] == migration.sha256(out / 'trajectories/00000.json')
    migration.migrate(out)
    assert json.loads((out / 'manifest.json').read_text()) == after


@pytest.mark.parametrize('change', ['code', 'questions', 'active', 'unknown_grader', 'bad_score', 'unknown_pipeline'])
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
    elif change == 'unknown_pipeline':
        manifest = json.loads((out / 'manifest.json').read_text())
        manifest['files']['scripts/frozen_critics.py'] = 'unknown'
        migration.write_json(out / 'manifest.json', manifest)
    else:
        migration.write_json(out / 'trajectories/00000.json', dict(question_id='q0', responses=[
            dict(token_ids=[4], score=1, verifier_status='parse_failure')]))
    before = (out / 'manifest.json').read_bytes()
    with pytest.raises(ValueError):
        migration.migrate(out)
    assert (out / 'manifest.json').read_bytes() == before
    assert not (out / 'manifest.before-exclusion-fix.json').exists()


def test_second_migration_preserves_first_migration_history(tmp_path, monkeypatch):
    _, out = fixture_run(tmp_path, monkeypatch)
    manifest = json.loads((out / 'manifest.json').read_text())
    manifest['verifier']['parse_error_policy'] = migration.PARSE_ERROR_POLICY
    manifest['files'][migration.GRADER] = '01a394be454cf2bc563e4d5f786390eb85271110d777de0e18fd75c494698dc4'
    migration.write_json(out / 'manifest.json', manifest)
    for name in ('manifest.before-sympify-fix.json', 'parser-fix-migration.json'):
        (out / name).write_text('first migration history')
    migration.migrate(out)
    for name in ('manifest.before-sympify-fix.json', 'parser-fix-migration.json'):
        assert (out / name).read_text() == 'first migration history'
