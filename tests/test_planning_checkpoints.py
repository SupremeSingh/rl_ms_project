import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from fit_math_critics import bounded_questions, screening_questions
from math_rl.provenance import sha256
from planning_checkpoints import evaluate, plan_boundary, render_viewer

TOKENIZER = SimpleNamespace(decode=lambda ids, **kwargs: ''.join(chr(i) for i in ids))


def test_bounded_splits_and_screen_exclusion():
    rows = [dict(id=f'{s}/{l}/{i}', split=s, level=l, source_index=-1)
            for s in ('train', 'val', 'test') for l in (4, 5) for i in range(160)]
    selected = bounded_questions(rows)
    assert len(selected) == 450
    assert not {q['id'] for q in selected} & {q['id'] for q in screening_questions(rows)}
    for split, n in [('train', 300), ('val', 50), ('test', 100)]:
        for level in (4, 5):
            assert sum(q['split'] == split and q['level'] == level for q in selected) == n // 2
    assert [q['source_index'] for q in selected] == list(range(450))
    assert all(q['source_index'] == -1 for q in rows)
    assert bounded_questions(rows[::-1]) == selected
    with pytest.raises(ValueError):
        bounded_questions(rows[:10])


def test_plan_boundary_requires_unique_complete_heading_and_nonterminal_state():
    text = 'Try addition.\nSolution:\nCompute 2+2.'
    boundary = plan_boundary(TOKENIZER, list(map(ord, text)))
    assert text[:boundary] == 'Try addition.\nSolution:\n'
    assert plan_boundary(TOKENIZER, list(map(ord, 'Just solve directly.'))) is None
    assert plan_boundary(TOKENIZER, list(map(ord, 'Plan.\nSolution:\n'))) is None
    assert plan_boundary(TOKENIZER, list(map(ord, text + '\nSolution:\nAgain'))) is None


@pytest.mark.parametrize('staged', [False, True])
def test_checkpoint_evaluation_reads_causal_positions_and_keeps_excluded_visible(tmp_path, staged):
    for style in ('completion', 'plan'):
        run = tmp_path / f'{style}-2048'
        (run / 'fit').mkdir(parents=True)
        (run / 'data/features').mkdir(parents=True)
        (run / 'data/trajectories').mkdir()
        q = dict(id='q1', question='<script>bad()</script>', ground_truth='2', split='test')
        (run / 'data/questions.json').write_text(json.dumps(dict(questions=[q])))
        torch.save(dict(baseline=.5), run / 'data/normalization.pt')
        for method in ('lstd-0.99', 'ridge'):
            torch.save(dict(weights=torch.tensor([.01, 0.], dtype=torch.float64),
                mean=torch.tensor([0.]), scale=torch.tensor([1.])), run / f'fit/{method}.pt')
        text = ('Plan first.\nSolution:\n' if style == 'plan' else '') + 'a' * 80
        ids = list(map(ord, text))
        answers = [dict(response=text, token_ids=ids, score=1., verifier_status='correct', finish_reason='stop')
                   for _ in range(3)] + [dict(response='<script>alert(1)</script>', token_ids=[65], score=None,
                       verifier_status='parse_failure', finish_reason='stop')]
        if staged:
            from math_rl.staged_generation import rollout, END
            engine = SimpleNamespace(generate=lambda prompts, params: [SimpleNamespace(outputs=[
                SimpleNamespace(token_ids=list(map(ord, 'a' * 40 + END)), finish_reason='stop', stop_reason=END)])])
            tokenizer = SimpleNamespace(encode=lambda text, **kw: list(map(ord, text)), decode=TOKENIZER.decode)
            answers[:3] = [dict(rollout(engine, tokenizer, [], style == 'plan', 42, SimpleNamespace),
                               score=1., verifier_status='correct') for _ in range(3)]
            ids = answers[0]['token_ids']
        path = run / 'data/trajectories/00000.json'
        path.write_text(json.dumps(dict(question_id='q1', responses=answers)))
        count = len(ids) + 1
        features = torch.arange(count).float()[:, None]
        features[-1] = 99999  # Post-terminal feature must never enter a checkpoint metric.
        torch.save(dict(question_id='q1', response_indices=torch.tensor([0, 1, 2]),
            offsets=torch.tensor([0, count, 2*count, 3*count]), rewards=torch.ones(3),
            features=features.repeat(3, 1), trajectories_sha256=sha256(path)), run / 'data/features/00000.pt')
    evaluate(tmp_path, TOKENIZER)
    report = json.loads((tmp_path / 'checkpoints.json').read_text())
    assert report['plan_heading_detected'] == 3
    assert report['total_plan_answers'] == 4
    assert report['metrics']['plan/32/ridge']['mean_prediction'] == pytest.approx(.32)
    assert report['metrics']['plan/0/ridge']['brier'] == 1
    if not staged:
        assert 'plan/128/ridge' not in report['metrics']
    else:
        assert report['boundary_version'] == 'controller-injected-v1'
        assert report['stage_audit']['plan']['facts']['attempts'] == 3
        assert report['metrics']['plan/post_plan/ridge']['mean_prediction'] == pytest.approx(1.02)
    assert report['paired_plan_minus_completion']['32/ridge']['difference'] == pytest.approx(0)
    if staged:
        # Control has terminated at this length; never use its terminal feature.
        assert report['paired_plan_minus_completion']['matched_plan_length/ridge'] is None
    else:
        assert report['paired_plan_minus_completion']['matched_plan_length/ridge']['difference'] == pytest.approx(0)
    html = (tmp_path / 'responses.html').read_text()
    assert '<script>' not in html
    assert '&lt;script&gt;' in html
    assert 'parse_failure' in html


@pytest.mark.parametrize('heading', ['Solution: ', '**Solution:** ', '### Solution: ',
    '**Step-by-Step Solution:**\n', 'Step by step Solution: '])
def test_heading_variants(heading):
    text = 'We will add the quantities.\n' + heading + 'Now calculate.'
    boundary = plan_boundary(TOKENIZER, list(map(ord, text)))
    assert boundary is not None
    assert text[:boundary].endswith(heading)


def test_format_compliance_and_fenced_headings():
    from planning_checkpoints import format_audit
    text = ('What we know: Two quantities.\nWhat we will do: Add them.\n'
            'Solution: Add 1 and 2.\nFinal answer: 3')
    assert format_audit(text)['ordered_sections']
    assert not format_audit(text.replace('Final answer: 3', 'Final answer:'))['ordered_sections']
    assert plan_boundary(TOKENIZER, list(map(ord, 'Plan.\n```\nSolution: fake\n```'))) is None
    assert not format_audit('What we know:\nWhat we will do:\nSolution:\nFinal answer: 3')['ordered_sections']


def test_structured_prompt_shares_final_answer_contract():
    from math_rl.prompts import encode_structured
    tokenizer = SimpleNamespace(encode=lambda text, **kw: list(map(ord, text)))
    control, _ = encode_structured(tokenizer, 'Find x.', False)
    plan, _ = encode_structured(tokenizer, 'Find x.', True)
    assert 'What we know:' in plan and 'What we will do:' in plan
    assert 'What we know:' not in control
    contract = r'Final answer: \boxed{number}'
    assert contract in control and contract in plan


def test_baseline_adjustment_removes_constant_difficulty_difference():
    from planning_checkpoints import paired
    left = [dict(question='q1', y=1., p=.4, baseline=.4)]
    right = [dict(question='q1', y=1., p=.8, baseline=.8)]
    assert paired(left, right)['difference'] == pytest.approx(.32)
    assert paired(left, right, adjusted=True)['difference'] == pytest.approx(0)
