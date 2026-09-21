import pytest
pytest.importorskip('math_verify')
from math_rl.math_verify_reward import grade, provenance, MAX_CHARS


@pytest.mark.parametrize('response,target,reward', [
    (r'The answer is \boxed{4}.', '4', 1),
    ('Therefore, Dad vacuumed upstairs for 2(11) + 5 = 27 minutes.', '27', 1),
    ('Answer: Philip will have a total of 80 paintings after 30 days.', '80', 1),
    (r'\boxed{(\frac{3000}{1500}) = 2 \text{ dollars/sq ft}}' + '\nThe cost is ' + r'\boxed{2}.', '2', 1),
    ('Therefore, Smith’s Bakery sold 70 pies.', '70', 1),
    ('The salesman’s profit is $222.', '442', 0),
    (r'\boxed{\frac{1}{2}}', '0.5', 1),
    (r'\boxed{17} then \boxed{60}', '17', 0),
])
def test_real_library(response, target, reward):
    result = grade(response, target)
    assert result['diagnostic_reward'] == reward
    assert result['status'] in ('correct', 'incorrect')


def test_failures_and_pins(monkeypatch):
    import math_verify
    from math_verify.errors import TimeoutException
    assert provenance()['packages']['math-verify'] == '0.9.0'
    assert grade('', '4')['status'] == 'parse_failure'
    assert grade('x' * (MAX_CHARS + 1), '4')['status'] == 'input_too_long'
    with pytest.raises(ValueError):
        grade('4', 'invalid')
    def timeout(*args, **kwargs):
        raise TimeoutException('test')
    monkeypatch.setattr(math_verify, 'parse', timeout)
    assert grade('4', '4')['status'] == 'parse_timeout'


@pytest.mark.parametrize('original_reward', [0, None])
def test_rescore_and_saved_labels_cli(tmp_path, original_reward):
    import hashlib
    import json
    import subprocess
    import sys
    from pathlib import Path
    rows = [dict(id=str(i), prompt_id='p', prompt='Question', response=text,
                 ground_truth='4', reward=original_reward, finish_reason='stop')
            for i, text in enumerate([r'The answer is \boxed{4}.', r'The answer is \boxed{5}.'])]
    payload = '\n'.join(json.dumps(r) for r in rows).encode()
    digest = hashlib.sha256(payload).hexdigest()
    (tmp_path / 'responses.jsonl').write_bytes(payload)
    (tmp_path / 'metadata.json').write_text(json.dumps({'responses_sha256': digest}))
    labels = '\n'.join(json.dumps(dict(id=str(i), unambiguous=1, correct=1-i,
                                      responses_sha256=digest)) for i in range(2))
    (tmp_path / 'final-numeric-v2-human.jsonl').write_text(labels)
    scripts = Path(__file__).parents[1] / 'scripts'
    for script, extra in [('rescore_stage1.py', []), ('review_final_answers.py', ['--report'])]:
        result = subprocess.run([sys.executable, str(scripts / script), str(tmp_path),
                                 '--rule', 'math-verify-v1', *extra], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / 'math-verify-v1-human-report.json').read_text())
    assert report['verifier_agreement'] == 1
    assert not report['stage1_pass']
    assert (tmp_path / 'responses.jsonl').read_bytes() == payload
    assert (tmp_path / 'final-numeric-v2-human.jsonl').read_text() == labels


def test_verification_timeout_is_not_a_wrong_answer(monkeypatch):
    import math_verify
    from math_verify.errors import TimeoutException
    def timeout(*args, **kwargs):
        raise TimeoutException('test')
    monkeypatch.setattr(math_verify, 'verify', timeout)
    result = grade(r'The answer is \boxed{4}.', '4')
    assert result['status'] == 'verify_timeout'
    assert result['diagnostic_reward'] == 0
    assert result['extracted'] == '4'
    from math_rl.verifier_batch import score_batch
    assert score_batch([dict(response=r'\boxed{4}', ground_truth='4')]) == [
        dict(score=0, verifier_status='verify_timeout', extracted='4')]


def test_sympify_extraction_error_is_unparseable_but_other_errors_still_abort(monkeypatch):
    import math_verify
    from sympy import SympifyError
    from math_rl.verifier_batch import score_batch

    def malformed(*args, **kwargs):
        raise SympifyError('malformed generated expression')
    monkeypatch.setattr(math_verify, 'parse', malformed)
    result = grade('malformed generated expression', '4')
    assert result['status'] == 'parse_failure'
    assert result['error'] == 'SympifyError' and result['extracted'] is None
    scores = score_batch([dict(response='malformed generated expression', ground_truth='4')])
    assert scores == [dict(score=0, verifier_status='parse_failure', extracted='')]

    def broken(*args, **kwargs):
        raise RuntimeError('unexpected backend error')
    monkeypatch.setattr(math_verify, 'parse', broken)
    with pytest.raises(RuntimeError, match='parse_error'):
        score_batch([dict(response='4', ground_truth='4')])


def test_sympify_error_during_equivalence_check_still_aborts(monkeypatch):
    import math_verify
    from sympy import SympifyError
    from math_rl.verifier_batch import score_batch

    def broken(*args, **kwargs):
        raise SympifyError('verification failed after successful extraction')
    monkeypatch.setattr(math_verify, 'verify', broken)
    with pytest.raises(RuntimeError, match='verify_error'):
        score_batch([dict(response=r'\boxed{4}', ground_truth='4')])
