import importlib.util
from pathlib import Path

import pytest
from math_rl.prompts import prompt_for_contract
from math_rl.reward import extract_answer, score_contract, compute_score


@pytest.mark.parametrize('response,expected', [
    (r'There are \(\boxed{56}\) crayons.', '56'),
    (r'It is \boxed { -1,200.50 }.', '-1200.50'),
    (r'\boxed{4} and \boxed{24}', None),
    (r'\boxed{4} and \boxed{4}', None),
    (r'\boxed{4} and \boxed{', None),
    (r'\boxed{\frac{8}{2}}', None),
    (r'\boxed{NaN}', None),
    (r'\boxed{1,20}', None),
    ('Answer: 56', None),
])
def test_matches_saved_response_rescorer(response, expected):
    spec = importlib.util.spec_from_file_location(
        'rescore', Path(__file__).parents[1] / 'scripts/rescore_stage1.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    value = extract_answer(response, 'numeric-box-v1')
    assert value == module.extract(response)
    assert (str(value) if value is not None else None) == expected


def test_candidate_is_explicit_and_scores_wrong_answers_zero():
    messages = prompt_for_contract('Question', 'numeric-box-v1')
    assert 'line' not in messages[0]['content']
    assert messages[1]['content'] == 'Question'
    assert score_contract(r'There are \boxed{56} crayons.', '56', 'numeric-box-v1') == 1
    assert score_contract(r'There are \boxed{64} crayons.', '56', 'numeric-box-v1') == 0
    assert compute_score('gsm8k', r'There are \boxed{56} crayons.', '56') == 0
