import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import fit_math_critics as runner
from math_rl.ppo_reward import EXCLUSION_POLICY
from math_rl.provenance import sha256, write_json
from test_lstd import fixture_source


def test_reference_box_is_balanced_and_not_approximately_converted():
    assert runner.last_box(r'Work \boxed{3} then \boxed{-42.5}.') == '-42.5'
    assert runner.last_box(r'\boxed{\frac{1}{3}}') == r'\frac{1}{3}'
    assert runner.last_box(r'\boxed{2') == ''
    assert runner.last_box('No answer') == ''


def test_splits_exclude_transfer_and_duplicates_before_sampling():
    def row(i, split='train', **kw):
        return dict(unique_id=f'{split}/{i}', official_split=split, problem=f'Problem {i}',
                    answer='2', solution=r'\boxed{2}', level=4, subject='algebra') | kw
    rows = [row(i, level=4 + i % 2) for i in range(30)]
    rows += [row(i, 'test') for i in range(30, 40)]
    rows += [row(100, problem=' Problem 30 '), row(101, problem='Problem 1')]
    tokenizer = SimpleNamespace(encode=lambda *a, **kw: [1, 2])
    selected = runner.split_questions(rows, [dict(problem='Problem 31')], tokenizer)
    qs = selected['questions']
    assert qs == runner.split_questions(list(reversed(rows)), [dict(problem='Problem 31')], tokenizer)['questions']
    texts = [runner.wording(q['question']) for q in qs]
    assert len(texts) == len(set(texts))
    assert runner.wording('Problem 31') not in texts
    assert all(q['official_split'] == ('test' if q['split'] == 'test' else 'train') for q in qs)
    assert {q['level'] for q in qs if q['split'] == 'val'} == {4, 5}
    assert [q['source_index'] for q in qs] == list(range(len(qs)))
    assert selected['reserved_math500_questions'] == 1
    capped = runner.split_questions(rows, [], tokenizer, dict(train=5, val=2, test=3))
    assert capped['counts'] == dict(train=5, val=2, test=3)


def test_prepare_fit_report_and_resume_use_only_training_normalization(tmp_path):
    source = tmp_path / 'data'
    questions = fixture_source(source)
    (source / 'summary.json').unlink()
    for i, q in enumerate(questions):
        q.update(level=4 + i % 2, source_index=i, question=f'Question {i}', ground_truth='1')
        raw_path = source / 'trajectories' / f'{i:05d}.json'
        raw = json.loads(raw_path.read_text())
        for response in raw['responses']:
            response['response'] = 'Generated answer'
        write_json(raw_path, raw)
        path = source / 'features' / f'{i:05d}.pt'
        shard = torch.load(path, weights_only=True)
        shard['exclusion_policy'] = EXCLUSION_POLICY
        shard['trajectories_sha256'] = sha256(raw_path)
        torch.save(shard, path)
        probe = source / f"{q['split']}.pt"
        if probe.exists():
            probe.unlink()
    write_json(source / 'questions.json', dict(questions=questions))
    write_json(source / 'manifest.json', dict(config=dict(protocol='frozen-math-critics-v1',
        exclusion_policy=EXCLUSION_POLICY), questions_sha256=sha256(source / 'questions.json')))
    runner.prepare(source, questions)
    norm = torch.load(source / 'normalization.pt', weights_only=True)
    train = torch.load(source / 'train.pt', weights_only=True)
    torch.testing.assert_close(norm['mean'], train['x'].float().mean(0))
    manifest = dict(config=dict(alphas=[.001, .01], lambdas=[0., .999, 1.]))
    runner.fit_and_report(tmp_path, manifest, questions)
    report = json.loads((tmp_path / 'summary.json').read_text())
    assert report['training_transitions'] == 4
    assert report['difficulty']['4']['selected_questions'] == 1  # test only, never train
    assert report['difficulty']['5']['selected_questions'] == 0
    assert len((source / 'review.jsonl').read_text().splitlines()) == 3
    assert 'held out' in report['evaluation']
    runner.fit_and_report(tmp_path, manifest, questions)
    assert json.loads((tmp_path / 'summary.json').read_text())['training_transitions'] == 4


@pytest.mark.parametrize('label,expected', [('Level 4', 4), (' Level   5 ', 5),
    ('5', 5), (4, 4), ('Level 3', 3), ('Level ?', None), ('', None),
    (None, None), ('Level 6', None), (True, None), (4.5, None)])
def test_dataset_difficulty_normalization(label, expected):
    row = runner.normalize_row(dict(level=label, problem='Problem', solution=r'\boxed{2}'),
                               'geometry', 'train', 7)
    assert row['level'] == expected
    assert row['raw_level'] == label
    assert row['unique_id'] == 'train/geometry/7'


def test_unknown_difficulty_excluded_without_losing_known_levels():
    from math_hard import select
    rows = [runner.normalize_row(dict(level=level, problem=f'Problem {i}', solution=r'\boxed{2}'),
                                 'geometry', 'train', i)
            for i, level in enumerate(('Level ?', 'Level 4', 'Level 5', 'Level 2'))]
    tokenizer = SimpleNamespace(encode=lambda *a, **kw: [1])
    selected, info = select(rows, tokenizer)
    assert [q['level'] for q in selected] == [4, 5]
    assert len(info['excluded']) == 2
    assert info['excluded'][0]['level'] is None
