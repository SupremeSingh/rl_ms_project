import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import rescore_staged_answers as audit


def test_final_only_rescore_preserves_source_and_is_resumable(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    data = source / 'data'
    (data / 'trajectories').mkdir(parents=True)
    (data / 'questions.json').write_text(json.dumps(dict(questions=[dict(id='q', question='<script>x</script>', ground_truth='6')])))
    answers = [dict(response='Earlier unrelated 6. Final: ' + text, final_stage_text=text,
                    verifier_status='invalid_final_stage', score=None)
               for text in ('The answer is 6.', 'The answer is 32.', '')]
    path = data / 'trajectories/00000.json'
    path.write_text(json.dumps(dict(question_id='q', responses=answers)))
    original = path.read_bytes()
    monkeypatch.setattr(audit, 'verifier_command', lambda: ['python'])
    import subprocess
    monkeypatch.setattr(subprocess, 'check_output', lambda *a, **kw: '{}')
    calls = []
    def score(sources, texts, refs, **kwargs):
        calls.extend(texts)
        assert kwargs == {'exclude_errors': True}
        return [dict(score=1, verifier_status='correct', extracted='6'),
                dict(score=0, verifier_status='incorrect', extracted='32')]
    monkeypatch.setattr(audit, 'compute_score', score)
    out = tmp_path / 'audit'
    audit.run(source, out)
    assert calls == ['Final answer: The answer is 6.', 'Final answer: The answer is 32.']
    report = json.loads((out / 'summary.json').read_text())
    assert report['recovered_incorrect'] == report['recovered_correct'] == 1
    assert report['new_statuses']['empty_final_stage'] == 1
    assert path.read_bytes() == original
    assert '<script>' not in (out / 'responses.html').read_text()
    audit.run(source, out)
    assert len(calls) == 2


def test_potential_ambiguity_is_flagged_for_review():
    assert audit.audit_flags('The result is 2 or 3.')
    assert not audit.audit_flags('The result is 32.')
