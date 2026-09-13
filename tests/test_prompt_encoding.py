from math_rl.prompts import encode_completion, AUDIT_INSTRUCTION


class Tokenizer:
    def encode(self, text, add_special_tokens):
        assert not add_special_tokens
        return list(text.encode())


def test_exact_completion_prefix():
    text, ids = encode_completion(Tokenizer(), 'How many?')
    assert bytes(ids).decode() == text
    assert text == AUDIT_INSTRUCTION + '\n\nQuestion: How many?\n\nSolution:'
    assert '<assistant>' not in text
