"""Predeclared MATH-500 level 4/5 decimal-answer transfer subset."""
from collections import Counter
import json
import re
from math_rl.reward import parse_number
from math_rl.prompts import encode_completion

DATASET = 'HuggingFaceH4/MATH-500'
POLICY = 'Levels 4/5; no diagram markers; plain integer/decimal reference; prompt <=512 tokens; all eligible rows'


def select(rows, tokenizer):
    questions, excluded, seen = [], [], set()
    for row in rows:
        key = row['unique_id']
        if key in seen:
            raise ValueError('Duplicate MATH question ID')
        seen.add(key)
        problem = row['problem']
        reason = None
        if row['level'] not in (4, 5):
            reason = 'level_not_4_or_5'
        elif re.search(r'\[asy\]|\\includegraphics|<img|\bdiagram\b|\bfigure\b', problem, re.I):
            reason = 'diagram_or_figure'
        elif parse_number(row['answer']) is None:
            reason = 'not_plain_integer_or_decimal'
        prompt, tokens = encode_completion(tokenizer, problem)
        if reason is None and len(tokens) > 512:
            reason = 'prompt_over_512_tokens'
        if reason:
            excluded.append(dict(id=key, level=row['level'], reason=reason))
            continue
        questions.append(dict(id='math500/' + key, question=problem, prompt=prompt,
            prompt_token_ids=tokens, ground_truth=str(parse_number(row['answer'])),
            original_answer=row['answer'], reference_solution=row['solution'],
            level=row['level'], subject=row['subject'], split='test', source_index=len(questions)))
    if not questions:
        raise ValueError('No eligible MATH questions')
    return questions, dict(policy=POLICY, considered=len(seen), selected=len(questions),
        levels=dict(Counter(str(q['level']) for q in questions)), excluded=excluded)


def load_questions(out):
    """Resume uses saved questions; first run resolves and pins a dataset commit."""
    saved = out / 'manifest.json'
    if saved.exists():
        manifest = json.loads(saved.read_text())
        if manifest.get('protocol') != 'lstd-math-hard-v1':
            raise ValueError('Output belongs to another protocol')
        return manifest['questions'], manifest['dataset']
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    from frozen_critics import ROOT
    revision = HfApi().dataset_info(DATASET).sha
    rows = load_dataset(DATASET, split='test', revision=revision)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/qwen-math', local_files_only=True)
    questions, info = select(rows, tokenizer)
    return questions, dict(info, name=DATASET, revision=revision, split='test')


def supplement(out, questions, data, predictions):
    import numpy as np
    from analyze_lstd import metadata, paired
    from frozen_critics import atomic_json
    meta, _ = metadata(out, questions, data)
    levels, audit = {}, []
    for level in (4, 5):
        subset = [q for q in questions if q['level'] == level and q['split'] == 'test']
        ids = [q['source_index'] for q in subset]
        result = paired(meta, predictions['lstd'], predictions['ridge'], np.isin(meta['question'], ids))
        statuses, capped, total, mixed = Counter(), 0, 0, 0
        for q in subset:
            raw = json.loads((out / 'trajectories' / f"{q['source_index']:05d}.json").read_text())
            retained = []
            for i, r in enumerate(raw['responses']):
                statuses[r['verifier_status']] += 1
                capped += r['finish_reason'] == 'length'
                total += 1
                if r['verifier_status'] in ('correct', 'incorrect'):
                    retained.append(r['score'])
                audit.append(dict(question_id=q['id'], answer_index=i, level=level,
                    question=q['question'], reference=q['ground_truth'], **r))
            mixed += 0 in retained and 1 in retained
        result.update(selected_questions=len(subset), total_answers=total, statuses=dict(statuses),
            truncation_rate=capped / total if total else None,
            mixed_question_rate=mixed / len(subset) if subset else None)
        levels[str(level)] = result
    atomic_json(out / 'difficulty-summary.json', levels)
    with (out / 'review.jsonl').open('w') as handle:
        for row in audit:
            handle.write(json.dumps(row) + '\n')
    return levels
