import importlib.util
from pathlib import Path
import pytest
from math_rl.data import make_row

spec = importlib.util.spec_from_file_location('generate', Path(__file__).parents[1] / 'scripts/generate_stage1.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Tokenizer:
    def encode(self, text, add_special_tokens):
        assert not add_special_tokens
        return list(range(600 if 'OVERLONG' in text else 20))


def test_selects_unique_questions_and_preserves_ids():
    rows = [make_row(dict(question=q, answer='work #### 4'), i)
            for i, q in enumerate(['OVERLONG', 'How many?', 'How  many?', 'Another?'])]
    chosen, excluded = module.select_rows(rows, Tokenizer(), 2)
    assert excluded == 1
    assert [r['extra_info']['index'] for r, _, _ in chosen] == [1, 3]
    assert chosen[0][1].endswith('Question: How many?\n\nSolution:')
    with pytest.raises(ValueError, match='need 3'):
        module.select_rows(rows, Tokenizer(), 3)
