"""Exercise the batch failure boundary without requiring the isolated parser install."""
import sys
from types import ModuleType

import pytest
from sympy import Rational, SympifyError

from math_rl import math_verify_reward as reward
from math_rl import verifier_batch


@pytest.mark.parametrize('failure,expected', [
    ('extraction_sympify', None), ('unexpected_parse', 'parse_error'),
    ('timeout', 'parse_timeout'), ('verification_sympify', 'verify_error'),
])
def test_only_known_extraction_conversion_error_is_nonfatal(monkeypatch, failure, expected):
    class TimeoutException(Exception):
        pass

    parser = ModuleType('math_verify')
    errors = ModuleType('math_verify.errors')
    errors.TimeoutException = TimeoutException

    def parse(*args, **kwargs):
        if failure == 'extraction_sympify':
            raise SympifyError('invalid expression')
        if failure == 'unexpected_parse':
            raise RuntimeError('broken parser')
        if failure == 'timeout':
            raise TimeoutException()
        return [Rational(4)]

    def verify(*args, **kwargs):
        raise SympifyError('unexpected equivalence failure')

    parser.parse, parser.verify = parse, verify
    monkeypatch.setitem(sys.modules, 'math_verify', parser)
    monkeypatch.setitem(sys.modules, 'math_verify.errors', errors)
    monkeypatch.setattr(reward, 'configuration', lambda: [])
    monkeypatch.setattr(verifier_batch, 'provenance', lambda: {})
    rows = [dict(response='generated expression', ground_truth='4')]
    if expected is None:
        assert verifier_batch.score_batch(rows) == [dict(score=0, verifier_status='parse_failure', extracted='')]
    else:
        with pytest.raises(RuntimeError, match=expected):
            verifier_batch.score_batch(rows)
