"""Planning changes prompts/budgets, never question identity or scoring contracts."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import math_planning
import frozen_critics
import fit_math_critics
import pytest
from math_rl.prompts import encode_completion, encode_planning


def test_screen_selection_is_balanced_validation_only_and_order_independent():
    questions = [dict(id=f'{split}/{level}/{i}', split=split, level=level)
                 for split in ('train', 'val', 'test') for level in (4, 5) for i in range(60)]
    selected = fit_math_critics.screening_questions(questions)
    assert len(selected) == 100
    assert all(q['split'] == 'val' for q in selected)
    assert sum(q['level'] == 4 for q in selected) == 50
    assert selected == fit_math_critics.screening_questions(list(reversed(questions)))
    with pytest.raises(ValueError, match='50 validation'):
        fit_math_critics.screening_questions(questions[:30])


def test_screen_only_launches_generation(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(sys, 'argv', ['fit_math_critics.py', '--out', str(tmp_path), '--screen'])
    monkeypatch.setattr(fit_math_critics, 'initialize', lambda *args: None)
    monkeypatch.setattr(fit_math_critics.subprocess, 'run',
                        lambda command, **kwargs: calls.append(command) or SimpleNamespace(returncode=0))
    fit_math_critics.main()
    assert len(calls) == 1
    assert calls[0][-2:] == ['--stage', 'generate']
    assert not (tmp_path / 'features').exists()


def test_screen_report_counts_failures_and_saves_paired_review(tmp_path):
    for style in ('completion', 'plan'):
        data = tmp_path / f'{style}-2048' / 'data'
        (data / 'trajectories').mkdir(parents=True)
        questions = [dict(id=f'q{level}', split='val', level=level,
                          question='Question', ground_truth='2') for level in (4, 5)]
        (data / 'questions.json').write_text(json.dumps(dict(questions=questions)))
        for i, q in enumerate(questions):
            statuses = ['correct', 'incorrect', 'parse_failure', 'verify_timeout']
            if style == 'plan':
                statuses[1] = 'correct'
            answers = [dict(response='Answer', verifier_status=status, token_ids=[1, 2], finish_reason='stop')
                       for status in statuses]
            (data / 'trajectories' / f'{i:05d}.json').write_text(json.dumps(
                dict(question_id=q['id'], responses=answers)))
    math_planning.summarize_screen(tmp_path)
    report = json.loads((tmp_path / 'summary.json').read_text())
    assert report['conditions']['completion']['answer_accuracy'] == .25
    assert report['conditions']['plan']['answer_accuracy'] == .5
    assert report['conditions']['plan']['verifier_statuses']['verify_timeout'] == 2
    assert report['plan_minus_completion']['accuracy_change_percentage_points'] == 25
    assert len((tmp_path / 'review.jsonl').read_text().splitlines()) == 2


def test_planning_conditions_and_encoding():
    assert math_planning.conditions() == [('completion', 2048), ('plan', 2048)]
    assert len(math_planning.conditions(True)) == 4
    tokenizer = SimpleNamespace(encode=lambda text, **kwargs: list(text.encode()))
    plain, _ = encode_completion(tokenizer, 'How many?')
    plan, tokens = encode_planning(tokenizer, 'How many?')
    assert 'brief plan of 2-3 sentences' in plan
    assert plan.endswith('Question: How many?\n\nSolution:')
    assert r'\boxed{...}' in plain and r'\boxed{...}' in plan
    assert bytes(tokens).decode() == plan


def test_generation_honors_response_budget_and_preserves_tokens(tmp_path, monkeypatch):
    calls = {}
    class Engine:
        def __init__(self, **kwargs):
            calls['engine'] = kwargs
        def generate(self, prompts, params):
            calls['params'] = params
            return [SimpleNamespace(outputs=[SimpleNamespace(text='2', token_ids=[2],
                finish_reason='stop', stop_reason=None)])]
    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(LLM=Engine, SamplingParams=lambda **kw: kw))
    import math_rl.ppo_reward
    monkeypatch.setattr(math_rl.ppo_reward, 'compute_score', lambda *a, **kw: [dict(score=1, verifier_status='correct')])
    (tmp_path / 'trajectories').mkdir()
    q = dict(id='math/test/1', prompt_token_ids=[1, 2, 3], ground_truth='2')
    frozen_critics.generate(tmp_path, dict(responses=1, sampling=dict(max_tokens=4096)), [q])
    assert calls['engine']['max_model_len'] == 4099
    assert calls['params']['max_tokens'] == 4096
    frozen_critics.generate(tmp_path, dict(responses=1, sampling=dict(max_tokens=4096)), [q])
    assert json.loads((tmp_path / 'trajectories/00000.json').read_text())['responses'][0]['token_ids'] == [2]


def test_report_separates_actor_accuracy_and_critic_error(tmp_path):
    for name, correct in [('completion-2048', False), ('plan-2048', True)]:
        run = tmp_path / name
        (run / 'data/trajectories').mkdir(parents=True)
        (run / 'fit').mkdir()
        (run / 'summary.json').write_text(json.dumps(dict(selected_lstd='lstd-0.99', methods={},
            paired_question_brier_selected_lstd_minus_ridge=0, paired_95pct_interval=[0, 0])))
        (run / 'fit/report.txt').write_text('Fixture critic report')
        (run / 'data/questions.json').write_text(json.dumps(dict(questions=[dict(id='q', split='test', source_index=0)])))
        (run / 'data/trajectories/00000.json').write_text(json.dumps(dict(responses=[
            dict(verifier_status='correct' if correct else 'incorrect', finish_reason='stop', token_ids=[1]),
            dict(verifier_status='parse_failure', finish_reason='length', token_ids=[2, 3]) ])))
        torch.save(dict(question=torch.tensor([0]), y=torch.tensor([float(correct)])), run / 'data/test.pt')
        for method in ('lstd-0.99', 'ridge'):
            np.save(run / f'fit/{method}-predictions.npy', np.array([.5]))
    math_planning.summarize(tmp_path, ['completion-2048', 'plan-2048'])
    report = json.loads((tmp_path / 'summary.json').read_text())
    assert report['conditions']['plan-2048']['answer_accuracy'] == .5
    assert report['conditions']['plan-2048']['retained'] == 1
    assert report['paired']['plan-2048']['answer_accuracy']['plan_minus_completion'] == .5
    assert report['paired']['plan-2048']['ridge']['plan_minus_completion'] == 0
