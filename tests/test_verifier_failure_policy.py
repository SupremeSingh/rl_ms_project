"""Exercise the batch failure boundary without requiring the isolated parser install."""
import sys
from types import ModuleType

import pytest
from sympy import Rational, SympifyError

from math_rl import math_verify_reward as reward
from math_rl import verifier_batch


@pytest.mark.parametrize('failure,expected', [
    ('extraction_sympify', None), ('unexpected_parse', 'parse_error'),
    ('timeout', None), ('verify_timeout', None), ('verification_sympify', 'verify_error'),
    ('parse_memory', 'MemoryError'), ('verify_memory', 'MemoryError'),
])
def test_only_known_extraction_conversion_error_is_nonfatal(monkeypatch, failure, expected):
    class TimeoutException(Exception):
        pass

    parser = ModuleType('math_verify')
    errors = ModuleType('math_verify.errors')
    errors.TimeoutException = TimeoutException

    def parse(*args, **kwargs):
        if failure == 'parse_memory':
            raise MemoryError()
        if failure == 'extraction_sympify':
            raise SympifyError('invalid expression')
        if failure == 'unexpected_parse':
            raise RuntimeError('broken parser')
        if failure == 'timeout':
            raise TimeoutException()
        return [Rational(4)]

    def verify(*args, **kwargs):
        if failure == 'verify_memory':
            raise MemoryError()
        if failure == 'verify_timeout':
            raise TimeoutException()
        raise SympifyError('unexpected equivalence failure')

    parser.parse, parser.verify = parse, verify
    monkeypatch.setitem(sys.modules, 'math_verify', parser)
    monkeypatch.setitem(sys.modules, 'math_verify.errors', errors)
    monkeypatch.setattr(reward, 'configuration', lambda: [])
    monkeypatch.setattr(verifier_batch, 'provenance', lambda: {})
    rows = [dict(response='generated expression', ground_truth='4')]
    if expected == 'MemoryError':
        for exclusion in (False, True):
            with pytest.raises(MemoryError):
                verifier_batch.score_batch(rows, exclude_errors=exclusion)
        return
    if expected is None:
        status = {'extraction_sympify': 'parse_failure', 'timeout': 'parse_timeout',
                  'verify_timeout': 'verify_timeout'}[failure]
        assert verifier_batch.score_batch(rows) == [dict(score=0, verifier_status=status,
                                                        extracted='4' if failure == 'verify_timeout' else '')]
    else:
        with pytest.raises(RuntimeError, match=expected):
            verifier_batch.score_batch(rows)
    excluded = verifier_batch.score_batch(rows, exclude_errors=True)
    assert excluded[0]['score'] is None
    assert excluded[0]['verifier_status'] in {'parse_failure', 'parse_error', 'parse_timeout', 'verify_timeout', 'verify_error'}


def test_exclusion_keeps_other_answers_and_does_not_hide_infrastructure(monkeypatch):
    monkeypatch.setattr(verifier_batch, 'provenance', lambda: {})
    def grade(text, gold):
        if text == 'bad':
            raise ValueError('unexpected expression conversion error')
        if text == 'resource':
            raise MemoryError()
        return dict(diagnostic_reward=int(text == 'right'), status='correct' if text == 'right' else 'incorrect', extracted='4')
    monkeypatch.setattr(verifier_batch, 'grade', grade)
    rows = [dict(response=text, ground_truth='4') for text in ('right', 'bad', 'wrong')]
    assert [r['score'] for r in verifier_batch.score_batch(rows, exclude_errors=True)] == [1, None, 0]
    with pytest.raises(MemoryError):
        verifier_batch.score_batch([dict(response='resource', ground_truth='4')], exclude_errors=True)
    with pytest.raises(ValueError, match='Invalid reference'):
        verifier_batch.score_batch([dict(response='right', ground_truth='invalid')], exclude_errors=True)


def test_worker_timeout_isolated_to_one_response(monkeypatch):
    import json
    import subprocess
    from types import SimpleNamespace
    from math_rl import ppo_reward
    monkeypatch.setattr(ppo_reward, 'verifier_command', lambda: ['python', '-m', 'fake'])
    def run(command, **kwargs):
        rows = json.loads(kwargs['input'])
        if len(rows) > 1 or rows[0]['response'] == 'slow':
            raise subprocess.TimeoutExpired(command, 23)
        return SimpleNamespace(stdout=json.dumps([dict(score=1, verifier_status='correct', extracted='4')]))
    monkeypatch.setattr(ppo_reward.subprocess, 'run', run)
    result = ppo_reward.compute_score(['gsm8k'] * 2, ['good', 'slow'], ['4'] * 2, exclude_errors=True)
    assert [r['score'] for r in result] == [1, None]
    assert result[1]['verifier_status'] == 'worker_timeout'
    with pytest.raises(subprocess.TimeoutExpired):
        ppo_reward.compute_score(['gsm8k'], ['slow'], ['4'])
