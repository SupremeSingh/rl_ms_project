import pytest
pytest.importorskip('math_verify')
from math_rl.conclusion_reward import grade_conclusion, select_conclusion, RULE
from math_rl.verifier_batch import score_batch

BASE = (r'Solving this equation, we find that $b = 7$ is the base that satisfies '
        r'the equation, where multiplication of $44$ and $55$ equals $3506$.')


def test_observed_base_conversion_miss_and_audit():
    result = grade_conclusion(BASE, '7')
    assert result['legacy_extracted'] == '3506'
    assert result['extracted'] == '7' and result['diagnostic_reward'] == 1
    assert result['selection_route'] == 'concluding_numeric_assignment'
    # Selection must not search for gold; reference changes never change extraction.
    wrong = grade_conclusion(BASE, '3506')
    assert wrong['extracted'] == '7' and wrong['diagnostic_reward'] == 0


@pytest.mark.parametrize('text,gold,expected', [
    ('Final answer: 19', '23', 0),
    (r'Final answer: \boxed{23}', '23', 1),
    (r'The reference might be 23. Final answer: \boxed{19}', '23', 0),
    ('```output\n27\n```\nFinal answer: 30', '30', 1),
    (r'We find that $b = -7$ is the base.', '-7', 1),
    (r'Therefore \(x = 0.5\) is the result.', '0.5', 1),
])
def test_explicit_conclusions(text, gold, expected):
    assert grade_conclusion(text, gold)['diagnostic_reward'] == expected


def test_ambiguity_and_malformed_equations_do_not_get_repaired():
    result = grade_conclusion(r'We find that $b = 7$, however the answer is 8.', '7')
    assert result['status'] == 'ambiguous_conclusion'
    candidate, _ = select_conclusion(r'We find that $b = 7 + 2$ is the answer.')
    assert candidate is None
    candidate, _ = select_conclusion('```python\nwe find that $b = 7$\n```')
    assert candidate is None


def test_rule_opt_in_and_batch_audit():
    row = dict(response=BASE, ground_truth='7')
    assert score_batch([row], exclude_errors=True)[0]['score'] == 0
    result = score_batch([dict(row, rule=RULE)], exclude_errors=True)[0]
    assert result['score'] == 1 and result['audit']['legacy_reward'] == 0
    with pytest.raises(ValueError, match='Unknown verifier'):
        score_batch([dict(row, rule='invented')], exclude_errors=True)


def test_multiple_final_answers_abstain_and_unclosed_code_is_ignored():
    result = grade_conclusion('Final answer: 7\nFinal answer: 8', '8')
    assert result['status'] == 'ambiguous_conclusion'
    assert select_conclusion('```python\nFinal answer: 7')[0] is None
