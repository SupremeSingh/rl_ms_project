import pytest
from math_rl.final_answer_v3 import extract_final_number


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


@pytest.mark.parametrize("response,expected", [
    (r"There are 8 pints in a gallon." + "\nThus, the vampire needs " + r"\boxed{4}" + " people.", "4"),
    (r"Therefore, the servings Georgie can make is \boxed{3}.", "3"),
    (r"Downstairs is 11 minutes. So, the answer is \boxed{27} minutes.", "27"),
    (r"The cost is \( \boxed{2} \). So, Tony is spending $2 per square foot.", "2"),
    ("Thus, the final answer is:\n" + r"\[" + "\n" + r"\boxed{10}" + "\n" + r"\]", "10"),
    (r"This is 2 hours and 30 minutes, or $\boxed{150}$ minutes.", "150"),
    (r"The result is \boxed{39}." + "\nWe add 18 + 21 = 39 months.", "39"),
    (r"Juniper has \boxed{6}." + "\nLet's confirm with code.\n```python\nprint(6)\n```", "6"),
    (r"\boxed{8}" + "\nHow many bracelets must you sell?\nThe answer is 50.", None),
    (r"\boxed{8}" + "\nThe total is 50.", None),
])
def test_reported_regressions(response, expected):
    value = extract_final_number(response)
    assert (str(value) if value is not None else None) == expected


@pytest.mark.parametrize("text,expected", [
    ("To ensure the answer is accurate and complete, let's break down the steps.\n"
     r"The vampire needs \boxed{4} people each day.", "4"),
    ("To verify that the answer is accurate, calculate again.\n"
     r"The answer is \boxed{4}.", "4"),
    ("The answer is 36 balls\nStep-by-step explanation:\n"
     "6 blue balls +18 green balls +4 red balls +8 yellow balls = 36 balls", "36"),
    ("The answer is 36 balls\nThe answer is 40 balls", None),
    ("The answer is 36 balls\nThe total is 40 balls", None),
    ("The answer is 36 balls\nQuestion: How many apples?\n2 + 2 = 4", None),
    ("The answer is 36 or 40 balls\nExplanation: 6 + 4 = 10", None),
    ("To ensure the answer is accurate, calculate again.", None),
])
def test_review_regressions(text, expected):
    value = extract_final_number(text)
    assert (str(value) if value is not None else None) == expected
