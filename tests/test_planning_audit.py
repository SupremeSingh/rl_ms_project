import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import audit_planning


def test_audit_preserves_sources_and_can_restart(tmp_path, monkeypatch):
    source, out = tmp_path / 'source', tmp_path / 'audit'
    for style in ('completion', 'plan'):
        data = source / f'{style}-2048/data'
        (data / 'trajectories').mkdir(parents=True)
        (data / 'questions.json').write_text(json.dumps(dict(questions=[
            dict(id='q', split='test', question='base?', ground_truth='7') ])))
        (data / 'trajectories/00000.json').write_text(json.dumps(dict(question_id='q', responses=[
            dict(response='Plan.\nSolution: b=7.', verifier_status='incorrect', score=0, extracted='3506') ])))
    original = {p: p.read_bytes() for p in source.rglob('*.json')}
    monkeypatch.setattr(audit_planning, 'compute_score', lambda sources, texts, gold, **kw:
                        [dict(score=1, verifier_status='correct', extracted='7') for _ in texts])
    audit_planning.audit(source, out)
    audit_planning.audit(source, out)
    assert all(p.read_bytes() == content for p, content in original.items())
    report = json.loads((out / 'summary.json').read_text())
    assert report['changed'] == 2
    assert report['totals']['plan']['new_correct'] == 1
    assert report['totals']['completion']['new_correct'] == 1
    assert report['totals']['plan']['new_unique_heading'] == 1
    assert report['totals']['plan']['old_unique_heading'] == 0
