import pytest
from math_rl.reward import compute_score


@pytest.mark.parametrize("response,target,expected", [
    ("Work...\nAnswer: 42", "42", 1.0),
    ("Answer: 41", "42", 0.0),
    ("Answer: 1,200.00", "1200", 1.0),
    ("Answer: -3.5", "-3.5", 1.0),
    ("", "42", 0.0),
    ("The answer is 42", "42", 0.0),
    ("Answer: 42 or 43", "42", 0.0),
    ("Answer: 42\nActually 43", "42", 0.0),
    ("Answer: 1,2,00", "1200", 0.0),
    ("Answer: NaN", "42", 0.0),
    ("Answer: 0.10000000000000001", "0.1", 0.0),
])
def test_score(response, target, expected):
    assert compute_score("gsm8k", response, target) == expected


def test_bad_reference_fails_loudly():
    with pytest.raises(ValueError):
        compute_score("gsm8k", "Answer: 42", "bad label")


@pytest.mark.parametrize("response,expected", [
    (r"\boxed{16}", 1.0),
    ("Work\n" + r"\boxed{16.0}", 1.0),
    (r"\boxed{17}", 0.0),
    (r"The answer is \boxed{16}.", 0.0),
    (r"\boxed{16}" + "\nMore", 0.0),
    (r"\boxed{17}" + "\n" + r"\boxed{16}", 0.0),
    ("Answer: 17\n" + r"\boxed{16}", 0.0),
    (r"\boxed{\frac{32}{2}}", 0.0),
    (r"\boxed{16", 0.0),
    (r"\boxed{NaN}", 0.0),
])
def test_boxed_contract(response, expected):
    from math_rl.reward import score_contract
    assert score_contract(response, "16", "boxed") == expected
