import pytest
from math_rl.final_answer import extract_final_number


@pytest.mark.parametrize('text,expected', [
    ('The answer is 60 minutes.', '60'),
    ('Therefore, Dad vacuumed upstairs for 27 minutes.', None),
    ('Answer: 27 minutes.', '27'),
    ('Therefore, Philip will have 80 paintings.', '80'),
    (r'The cost is \(\boxed{2}\) dollars per square foot.', '2'),
    (r'\boxed{2}' + '\n' + r'The answer is \boxed{2}.', '2'),
    ('The answer is 17.\nThe answer is 60.', None),
    (r'\boxed{17}' + '\nThe answer is 60.', None),
    ('The answer is 17 or 60.', None),
    ('The answer is approximately 60.', None),
    ('The answer is 1/2.', None),
    ('The answer is 1e3.', None),
    ('The answer is 1,20.', None),
    (r'\boxed{\frac{4}{2}}', None),
    (r'\boxed{2', None),
    ('```output\n60\n```', None),
    ('```python\nprint(60)', None),
    ('Some reasoning uses 60.', None),
    ('The answer is -1,200.50 dollars.', '-1200.50'),
])
def test_final_extractor(text, expected):
    value = extract_final_number(text)
    assert (str(value) if value is not None else None) == expected
