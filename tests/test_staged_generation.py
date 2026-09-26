from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from math_rl.staged_generation import END, decision_states, rollout, stage_audit, stages, validate


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(text.encode())

    def decode(self, ids, **kwargs):
        return bytes(ids).decode()


class Engine:
    def __init__(self, texts):
        self.texts = iter(texts)
        self.calls = []

    def generate(self, prompts, params):
        self.calls.append((deepcopy(prompts), params))
        text, finish = next(self.texts)
        output = SimpleNamespace(token_ids=list(text.encode()), finish_reason=finish,
                                 stop_reason=END if finish == 'stop' else None)
        return [SimpleNamespace(outputs=[output])]


@pytest.mark.parametrize('planning', [False, True])
def test_staged_rollout_exact_context_and_causal_features(planning):
    schedule = stages(planning)
    texts = [('some content' + END, 'stop')] * (len(schedule) - 1) + [('42' + END, 'stop')]
    engine = Engine(texts)
    row = rollout(engine, Tokenizer(), [1, 2], planning, 42, SimpleNamespace)
    assert sum(call[1].max_tokens for call in engine.calls) == 2048
    assert row['final_stage_text'] == '42'
    assert row['inserted_tokens'] > 0
    assert len(row['context_token_ids']) == len(row['token_ids']) + row['inserted_tokens']
    for (prompts, params), stage in zip(engine.calls, row['stages']):
        assert prompts[0]['prompt_token_ids'] == [1, 2] + row['context_token_ids'][:stage['action_start']]
        assert params.stop == [END]
        assert params.include_stop_str_in_output
        assert not any(row['action_mask'][stage['header_start']:stage['action_start']])
    all_states = torch.arange(len(row['context_token_ids']) + 1)[:, None]
    selected = decision_states(all_states, row)
    assert selected.shape[0] == len(row['token_ids']) + 1
    for stage in row['stages']:
        assert selected[stage['action_offset']].item() == stage['action_start']
    assert selected[-1].item() == len(row['context_token_ids'])
    assert all(v['attempts'] == 1 for v in stage_audit([row]).values())


def test_caps_empty_content_and_corrupt_action_masks():
    engine = Engine([('x' * 96, 'length'), (END, 'stop'), ('compute' + END, 'stop'), ('2' + END, 'stop')])
    row = rollout(engine, Tokenizer(), [], True, 42, SimpleNamespace)
    assert row['finish_reason'] == 'length'
    audit = stage_audit([row])
    assert audit['facts']['capped'] == 1
    assert audit['plan']['empty'] == 1
    broken = deepcopy(row)
    broken['action_mask'][0] = True
    with pytest.raises(ValueError):
        validate(broken)
    broken = deepcopy(row)
    broken['stages'][1]['action_offset'] += 1
    with pytest.raises(ValueError):
        validate(broken)
    with pytest.raises(ValueError):
        decision_states(torch.zeros(2, 1), row)


def test_generation_scores_only_final_stage_and_resumes(tmp_path, monkeypatch):
    import json
    import sys
    import transformers
    from math_rl import ppo_reward
    from math_rl.staged_generation import generate
    engine = Engine([('Incorrect intermediate number 99' + END, 'stop'), ('42' + END, 'stop')])
    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(LLM=lambda **kw: engine, SamplingParams=SimpleNamespace))
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda *a, **kw: Tokenizer())
    scored = []
    def score(sources, answers, golds, **kwargs):
        scored.extend(answers)
        return [dict(score=1., verifier_status='correct')]
    monkeypatch.setattr(ppo_reward, 'compute_score', score)
    (tmp_path / 'trajectories').mkdir()
    config = dict(prompt_style='completion', responses=1, generation_seed=42, verifier_rule='math-verify-conclusion-v2')
    questions = [dict(id='q', prompt_token_ids=[1], ground_truth='42')]
    save = lambda path, value: path.write_text(json.dumps(value))
    generate(tmp_path, config, questions, tmp_path, save)
    assert scored == ['Final answer: 42']
    generate(tmp_path, config, questions, tmp_path, save)
    assert len(engine.calls) == 2
    assert not list((tmp_path / 'pending').glob('*.json'))
