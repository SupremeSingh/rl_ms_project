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
