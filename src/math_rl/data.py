"""Pinned numeric Numina ingestion and outcome-independent question-family splits."""
from collections import Counter
import json
from pathlib import Path
import re

from slice_critic.config import immutable_revision
from slice_critic.references import reference_decision
from slice_critic.text import digest, family_key
from .prompts import encode_completion, make_prompt
from .provenance import sha256, write_json

DATASET = 'AI-MO/NuminaMath-1.5'
ROOT = Path(__file__).resolve().parents[2]
SUBJECTS = ('algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra',
            'number_theory', 'prealgebra', 'precalculus')


def candidate(row):
    if (row.get('source') != 'cn_k12' or row.get('synthetic') is not False
            or row.get('question_type') != 'math-word-problem'
            or row.get('problem_is_valid') != 'Yes' or row.get('solution_is_valid') != 'Yes'):
        return None, 'source_or_metadata'
    problem = row['problem'].strip()
    if re.search(r'!\[|\\includegraphics|https?://|\b(?:diagram|figure|shown below|prove that)\b', problem, re.I):
        return None, 'non_text_or_proof'
    decision = reference_decision(problem, row['solution'], str(row.get('answer', '')))
    if not decision['eligible']:
        return None, '|'.join(decision['reasons'])
    return dict(question=problem, ground_truth=decision['numeric_reference'],
                reference_solution=row['solution'], reference_audit=decision), None


def select_questions(rows, blocked, tokenizer, counts, salt='numina-online-v1'):
    eligible, reasons = {}, Counter()
    for index, row in enumerate(rows):
        q, reason = candidate(row)
        if q is None:
            reasons[reason] += 1
            continue
        family = family_key(q['question'])
        if family in blocked or family in eligible:
            reasons['overlap_or_duplicate'] += 1
            continue
        if not 0 < len(encode_completion(tokenizer, q['question'])[1]) <= 512:
            reasons['prompt_length'] += 1
            continue
        eligible[family] = dict(q, family=family, id=f'numina-cn-k12/{index}', source_index=index, level=0)
    ordered = sorted(eligible.values(), key=lambda q: digest([salt, q['family']]))
    if len(ordered) < sum(counts.values()):
        raise ValueError('Insufficient eligible families; filters and budgets are never silently relaxed')
    selected, offset = [], 0
    for split, count in counts.items():
        selected.extend(dict(q, split=split) for q in ordered[offset:offset+count])
        offset += count
    return selected, dict(eligible=len(eligible), exclusions=dict(reasons), counts=counts)


def write_online_data(out, questions):
    from datasets import Dataset
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    for split in ('train', 'val', 'test'):
        records = [dict(data_source='numina_numeric', prompt=make_prompt(q['question']), ability='math',
            reward_model=dict(style='rule', ground_truth=q['ground_truth']),
            extra_info=dict(index=q['source_index'], prompt_id=q['id'], split=split, level=0))
            for q in questions if q['split'] == split]
        if not records:
            raise ValueError(f'Empty {split} split')
        Dataset.from_list(records).to_parquet(str(out/f'{split}.parquet'))
    write_json(out/'questions.json', dict(questions=questions))


def prepare(out, model, revisions=None, offline_count=13000, counts=None):
    """New SLICE training corpus and disjoint RL data; not an exact historical filter rebuild."""
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    from slice_critic import TrainingConfig
    from dataclasses import asdict
    out, model = Path(out).resolve(), Path(model).resolve()
    out.mkdir(parents=True, exist_ok=False)
    names = (DATASET, 'openai/gsm8k', 'EleutherAI/hendrycks_math')
    revisions = revisions or {name: HfApi().dataset_info(name).sha for name in names}
    if set(revisions) != set(names) or not all(immutable_revision(v) for v in revisions.values()):
        raise ValueError('All three datasets need immutable revision SHAs')
    write_json(out/'revisions.json', revisions)
    blocked = set()
    for split in ('train', 'test'):
        blocked.update(family_key(q['question']) for q in load_dataset('openai/gsm8k', 'main',
            split=split, revision=revisions['openai/gsm8k']))
        for subject in SUBJECTS:
            blocked.update(family_key(q['problem']) for q in load_dataset('EleutherAI/hendrycks_math', subject,
                split=split, revision=revisions['EleutherAI/hendrycks_math']))
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    rows = load_dataset(DATASET, split='train', revision=revisions[DATASET])
    offline, offline_audit = select_questions(rows, blocked, tokenizer, {'offline': offline_count}, salt='slice-offline-v1')
    blocked.update(q['family'] for q in offline)
    online, online_audit = select_questions(rows, blocked, tokenizer,
        counts or dict(train=3000, val=500, test=2000))
    datafile = out/'critic.jsonl'
    datafile.write_text(''.join(json.dumps(dict(problem=q['question'], answer=q['ground_truth'],
        solution=q['reference_solution']))+'\n' for q in offline))
    write_online_data(out/'online', online)
    assets = json.loads((ROOT/'configs/assets.json').read_text())
    config = TrainingConfig(dataset=str(datafile), model_path=str(model), model_revision=assets['model_revision'], solution_column='solution',
        verifier_python=str(Path('.venv-ppo-verifier/bin/python').absolute()),
        max_questions=offline_count, responses_per_question=8, max_new_tokens=3584, epochs=120)
    write_json(out/'critic-config.json', asdict(config))
    write_json(out/'manifest.json', dict(protocol='slice-numina-data-v1', revisions=revisions,
        offline_audit=offline_audit, online_audit=online_audit,
        hashes={str(p.relative_to(out)): sha256(p) for p in out.rglob('*') if p.is_file()},
        scope='Fresh from-scratch representation run; reference-only selection, family-disjoint RL questions'))
