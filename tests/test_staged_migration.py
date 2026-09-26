import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import migrate_staged_tokens as migration
from math_rl.provenance import sha256
from test_staged_generation import Engine, Tokenizer
from math_rl.staged_generation import END, rollout
from types import SimpleNamespace


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(migration, 'VERSION', 'staged-plan-v1')
    root = tmp_path / 'repo'
    code = root / migration.CODE
    code.parent.mkdir(parents=True)
    code.write_text('patched code')
    other = root / 'other.py'
    other.write_text('unchanged')
    model = root / 'models/qwen-math'
    model.mkdir(parents=True)
    (model / 'tokenizer.json').write_text(json.dumps(dict(model=dict(vocab=Tokenizer().get_vocab()))))
    monkeypatch.setattr(migration, 'ROOT', root)
    out = root / 'outputs/run'
    run = out / 'completion-2048'
    data = run / 'data'
    (data / 'trajectories').mkdir(parents=True)
    (data / 'pending').mkdir()
    (data / 'questions.json').write_text(json.dumps(dict(questions=[dict(id='q', prompt_token_ids=[1]) ])))
    manifest = dict(config=dict(multistage=True, responses=2), files={str(code): migration.OLD_SHA, str(other): sha256(other)})
    (run / 'manifest.json').write_text(json.dumps(manifest))
    (data / 'manifest.json').write_text(json.dumps(dict(manifest, questions_sha256=sha256(data / 'questions.json'))))
    (run / 'status.json').write_text(json.dumps(dict(stage='generate', state='failed')))
    engine = Engine([('x' + END, 'stop'), ('2' + END, 'stop')])
    row = dict(rollout(engine, Tokenizer(), [1], False, 42, SimpleNamespace), score=1, verifier_status='correct')
    bad = json.loads(json.dumps(row))
    bad['context_token_ids'][-1] = bad['token_ids'][-1] = 151779
    path = data / 'trajectories/00000.json'
    path.write_text(json.dumps(dict(question_id='q', responses=[row, bad])))
    return out, path, row, other


def test_recovery_preserves_valid_answer_and_records_exclusion(tmp_path, monkeypatch):
    out, path, row, _ = fixture(tmp_path, monkeypatch)
    before = path.read_bytes()
    migration.migrate(out)
    after = json.loads(path.read_text())['responses']
    assert after[0] == row
    assert after[1]['score'] is None
    assert after[1]['verifier_status'] == 'invalid_token_id'
    assert (out / 'token-guard-backup/completion-2048/data/trajectories/00000.json').read_bytes() == before
    assert json.loads((out / 'token-guard-migration.json').read_text())['excluded'] == 1
    migrated = path.read_bytes()
    migration.migrate(out)
    assert path.read_bytes() == migrated


def test_recovery_refuses_unrelated_changes_without_writes(tmp_path, monkeypatch):
    out, path, _, other = fixture(tmp_path, monkeypatch)
    before = path.read_bytes()
    other.write_text('changed')
    with pytest.raises(ValueError, match='Unrelated'):
        migration.migrate(out)
    assert path.read_bytes() == before
    assert not (out / 'token-guard-backup').exists()
