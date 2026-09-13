import pytest
from math_rl.prompts import encode_prompt, prompt_for_contract


class Tokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert add_generation_prompt
        text = '<system>' + messages[0]['content'] + '<user>' + messages[1]['content'] + '<assistant>'
        return list(text.encode()) if tokenize else text

    def encode(self, text, add_special_tokens):
        assert not add_special_tokens
        return list(text.encode())


def test_exact_prefix_and_same_content():
    messages = prompt_for_contract('How many?', 'numeric-box-v1')
    tokenizer = Tokenizer()
    for style in ('chat', 'completion'):
        text, ids = encode_prompt(tokenizer, messages, style)
        assert bytes(ids).decode() == text
        assert messages[0]['content'] in text
        assert messages[1]['content'] in text
    text, _ = encode_prompt(tokenizer, messages, 'completion')
    assert text.endswith('\n\nSolution:')
    assert '<assistant>' not in text
    with pytest.raises(ValueError):
        encode_prompt(tokenizer, messages, 'unknown')


def test_example_is_only_in_prefix():
    from math_rl.reward import score_contract
    messages = prompt_for_contract('A different question?', 'numeric-box-v1')
    text, ids = encode_prompt(Tokenizer(), messages, 'completion-example-v1')
    assert bytes(ids).decode() == text
    assert text.count('Question:') == 2
    assert text.endswith('Question: A different question?\n\nSolution:')
    assert r"\boxed{7}" in text
    # Score only generated response text, never the demonstration in the prefix.
    assert score_contract('The answer is 7.', '7', 'numeric-box-v1') == 0
    assert score_contract(r'The answer is \boxed{3}.', '3', 'numeric-box-v1') == 1
