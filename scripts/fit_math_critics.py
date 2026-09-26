"""Independent MATH level 4/5 data collection and matched LSTD/ridge fitting."""
import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from frozen_critics import atomic_json, extract, generate, prepare_probes, save_tensor
from math_hard import DATASET, select, supplement
from math_rl.critic_probe import PREFIX_SAMPLING
from math_rl.ppo_reward import EXCLUSION_POLICY, verifier_command
from math_rl.provenance import sha256

FULL_DATASET = 'EleutherAI/hendrycks_math'
SUBJECTS = ('algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra',
            'number_theory', 'prealgebra', 'precalculus')
LIMITS = {'train': 1000, 'val': 200, 'test': 300}


def wording(text):
    return ''.join(unicodedata.normalize('NFKC', text).split()).casefold()


def last_box(solution):
    """Extract the last balanced reference box; numeric eligibility is checked separately."""
    start = solution.rfind('\\boxed{')
    if start < 0:
        return ''
    start += len('\\boxed{')
    depth = 1
    for i in range(start, len(solution)):
        depth += (solution[i] == '{') - (solution[i] == '}')
        if depth == 0:
            return solution[start:i]
    return ''


def split_questions(rows, transfer_rows, tokenizer, limits=LIMITS):
    """Preserve official test; stratify validation by level/subject, before generation."""
    blocked = {wording(q['problem']) for q in transfer_rows}
    official_test = {wording(q['problem']) for q in rows if q['official_split'] == 'test'}
    filtered, exclusions, seen = [], [], set()
    for row in sorted(rows, key=lambda q: q['unique_id']):
        text = wording(row['problem'])
        reason = None
        if text in blocked:
            reason = 'reserved_math500'
        elif row['official_split'] == 'train' and text in official_test:
            reason = 'duplicate_of_official_test'
        elif text in seen:
            reason = 'duplicate_wording'
        if reason:
            exclusions.append(dict(id=row['unique_id'], reason=reason))
            continue
        seen.add(text)
        filtered.append(row)
    questions, selection = select(filtered, tokenizer)
    originals = {q['unique_id']: q for q in filtered}
    for q in questions:
        key = q['id'].removeprefix('math500/')
        q['id'] = 'math/' + key
        q['official_split'] = originals[key]['official_split']
    def order(q):
        return hashlib.sha256(('math-split-271828:' + q['id']).encode()).hexdigest()
    strata = {}
    test = []
    for q in questions:
        if q['official_split'] == 'test':
            test.append(q)
        else:
            strata.setdefault((q['level'], q['subject']), []).append(q)
    pools = dict(train=[], val=[], test=test)
    for group in strata.values():
        group.sort(key=order)
        n = max(1, round(len(group) * .2)) if len(group) > 1 else 0
        pools['val'].extend(group[:n])
        pools['train'].extend(group[n:])
    selected = []
    for split, pool in pools.items():
        pool.sort(key=order)
        for q in pool[limits[split]:]:
            exclusions.append(dict(id=q['id'], reason='predeclared_' + split + '_limit'))
        for q in pool[:limits[split]]:
            q.update(split=split, source_index=len(selected))
            selected.append(q)
        if not pool:
            raise ValueError(f'No eligible {split} questions')
    return dict(questions=selected, selection=selection, exclusions=exclusions,
        counts=dict(Counter(q['split'] for q in selected)), limits=limits,
        split_rule='Official train: 80/20 within level/subject, stable SHA256 order seed 271828; official test held out; fixed caps',
        reserved_math500_questions=len(blocked))


def normalize_row(row, subject, split, index):
    """Unknown difficulty stays unknown; never infer it from the problem or answer."""
    raw_level = row.get('level')
    if type(raw_level) is int and 1 <= raw_level <= 5:
        level = raw_level
    elif isinstance(raw_level, str):
        match = re.fullmatch(r'(?:Level\s+)?([1-5])', raw_level.strip())
        level = int(match[1]) if match else None
    else:
        level = None
    return dict(row, unique_id=f'{split}/{subject}/{index}', official_split=split,
                subject=subject, level=level, raw_level=raw_level,
                answer=last_box(row['solution']))


def download_questions():
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    revisions = {name: HfApi().dataset_info(name).sha for name in (FULL_DATASET, DATASET)}
    reserved = list(load_dataset(DATASET, revision=revisions[DATASET], split='test'))
    rows = []
    for subject in SUBJECTS:
        for split in ('train', 'test'):
            dataset = load_dataset(FULL_DATASET, subject, split=split, revision=revisions[FULL_DATASET])
            for i, row in enumerate(dataset):
                rows.append(normalize_row(row, subject, split, i))
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/qwen-math', local_files_only=True)
    selected = split_questions(rows, reserved, tokenizer)
    selected['unknown_difficulty'] = [dict(id=r['unique_id'], raw_level=r['raw_level'])
                                      for r in rows if r['level'] is None]
    print(f"Excluded {len(selected['unknown_difficulty'])} rows with unknown difficulty labels", flush=True)
    selected['dataset_revisions'] = revisions
    return selected


def screening_questions(questions):
    """Fixed, outcome-independent validation sample: 50 questions per level."""
    selected = []
    for level in (4, 5):
        pool = [q for q in questions if q['split'] == 'val' and q['level'] == level]
        pool.sort(key=lambda q: hashlib.sha256(('planning-screen-42:' + q['id']).encode()).hexdigest())
        if len(pool) < 50:
            raise ValueError(f'Screen requires 50 validation questions at level {level}')
        selected.extend(pool[:50])
    return selected


def bounded_questions(questions):
    """Preserve splits and exclude the deterministic screening sample."""
    blocked = {q['id'] for q in screening_questions(questions)}
    selected = []
    for split, count in (('train', 300), ('val', 50), ('test', 100)):
        for level in (4, 5):
            pool = [q for q in questions if q['split'] == split and q['level'] == level
                    and q['id'] not in blocked]
            pool.sort(key=lambda q: hashlib.sha256(('planning-critic-42:' + q['id']).encode()).hexdigest())
            if len(pool) < count // 2:
                raise ValueError(f'Not enough unscreened {split} level {level} questions')
            selected.extend(dict(q) for q in pool[:count // 2])
    for index, q in enumerate(selected):
        q['source_index'] = index
    return selected


def initialize(out, source_run=None, prompt_style="completion", budget=2048, screen=False, bounded=False, structured=False, multistage=False):
    config = dict(protocol='frozen-math-critics-v1', responses=16, generation_seed=161803,
        data_source='math_numeric', exclusion_policy=EXCLUSION_POLICY, prefix_sampling=PREFIX_SAMPLING,
        sampling=dict(temperature=1., top_p=1., top_k=-1, max_tokens=2048, seed='161803 + question index'),
        limits=LIMITS, lambdas=[0., .9, .99, .999, 1.], alphas=[0., 1e-6, 1e-5, 1e-4, .001, .01, .1, 1.])
    if source_run is not None:
        source_run = source_run.resolve()
        if not (source_run / 'report.txt').exists():
            raise ValueError('Source must be a completed MATH fit')
        source_data = source_run / 'data'
        original = json.loads((source_data / 'manifest.json').read_text())
        if original['questions_sha256'] != sha256(source_data / 'questions.json'):
            raise ValueError('Source question snapshot changed')
        if original['config']['protocol'] != 'frozen-math-critics-v1':
            raise ValueError('Expected a MATH fitting source')
        config.update(prompt_style=prompt_style, source_run=str(source_run),
                      source_questions_sha256=sha256(source_data / 'questions.json'), lambdas=[.99])
        config['sampling']['max_tokens'] = budget
    elif prompt_style != 'completion' or budget != 2048:
        raise ValueError('Planning experiments require --source-run')
    if screen:
        if source_run is None or budget != 2048:
            raise ValueError('Screen requires a source run and 2048-token budget')
        config.update(screen=True, responses=4)
    if bounded:
        if screen or source_run is None or budget != 2048:
            raise ValueError('Bounded critics require a source, 2048 tokens, and no screen flag')
        config.update(bounded=True, responses=4, limits=dict(train=300, val=50, test=100),
                      alphas=[1e-4, .001, .01, .1, 1.], plan_boundary='explicit Solution heading v1')
    if structured:
        if not bounded:
            raise ValueError('Structured protocol requires bounded fitting')
        config.update(prompt_protocol='known-actions-v2', verifier_rule='math-verify-conclusion-v2',
                      plan_boundary='explicit-headings-v2')
    if multistage:
        if not structured:
            raise ValueError('Multistage requires structured bounded fitting')
        from math_rl.staged_generation import stages, VERSION
        config.update(multistage=True, generation_protocol=VERSION,
            stage_schedule=[list(s) for s in stages(prompt_style == 'plan')],
            stage_seed='generation_seed + question_index * 100 + sample_index * 10 + stage_index',
            plan_boundary='controller-injected-v1',
            scoring_scope='final stage only; no credit from plan or solution numbers')
    paths = [ROOT / p for p in ('scripts/fit_math_critics.py', 'scripts/math_hard.py',
        'scripts/frozen_critics.py', 'scripts/fit_lstd.py', 'scripts/analyze_lstd.py',
        'src/math_rl/critic_probe.py', 'src/math_rl/lstd.py', 'src/math_rl/ppo_reward.py',
        'src/math_rl/verifier_batch.py', 'src/math_rl/math_verify_reward.py',
        'src/math_rl/reward.py', 'src/math_rl/prompts.py', 'src/math_rl/provenance.py')]
    if bounded:
        paths += [ROOT / 'scripts/planning_checkpoints.py', ROOT / 'scripts/math_planning.py']
    if structured:
        paths += [ROOT / 'src/math_rl/conclusion_reward.py']
    if multistage:
        paths += [ROOT / 'src/math_rl/staged_generation.py']
    model_files = sorted((ROOT / 'models/qwen-math').glob('*.safetensors'))
    if not model_files:
        raise ValueError('Missing local base model')
    if source_run is not None:
        for path in model_files + sorted((ROOT / 'models/qwen-math').glob('*.json')):
            expected = original['files'].get(str(path))
            if expected != sha256(path):
                raise ValueError(f'Source model differs: {path}')
        paths += [source_data / 'manifest.json', source_data / 'questions.json']
    paths += model_files + sorted((ROOT / 'models/qwen-math').glob('*.json'))
    verifier = json.loads(subprocess.check_output(verifier_command() + ['--provenance'], text=True,
        env=dict(os.environ, PYTHONPATH=str(ROOT / 'src'))))
    manifest = dict(config=config, files={str(p): sha256(p) for p in paths}, verifier=verifier,
        versions={p: version(p) for p in ('torch', 'transformers', 'vllm', 'datasets', 'numpy')})
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        previous = json.loads((out / 'manifest.json').read_text())
        if previous != manifest:
            raise ValueError('Resume requires unchanged code, model, settings, runtime and verifier')
    else:
        if any(out.iterdir()):
            raise ValueError('Choose an empty output directory')
        atomic_json(out / 'manifest.json', manifest)
    source = out / 'data'
    source.mkdir(exist_ok=True)
    if not (source / 'questions.json').exists():
        if source_run is None:
            selection = download_questions()
        else:
            from transformers import AutoTokenizer
            from math_rl.prompts import encode_completion, encode_planning, encode_plan_sections, encode_structured
            tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/qwen-math', local_files_only=True)
            selection = json.loads((source_data / 'questions.json').read_text())
            if screen:
                selection = dict(questions=screening_questions(selection['questions']),
                    counts={'val': 100}, split_rule='Validation only; SHA256 planning-screen-42; 50 per level')
            if bounded:
                selection = dict(questions=bounded_questions(selection['questions']), counts=config['limits'],
                    split_rule='Original splits, balanced levels, SHA256 planning-critic-42; screening sample excluded',
                    test_previously_inspected=True)
            encoder = encode_planning if prompt_style == 'plan' else encode_completion
            if bounded and prompt_style == 'plan':
                encoder = encode_plan_sections
            for q in selection['questions']:
                q['prompt'], q['prompt_token_ids'] = (encode_structured(tokenizer, q['question'], prompt_style == 'plan')
                    if structured else encoder(tokenizer, q['question']))
                if multistage:
                    from math_rl.staged_generation import prompt
                    q['prompt'], q['prompt_token_ids'] = prompt(tokenizer, q['question'], prompt_style == 'plan')
        atomic_json(source / 'questions.json', selection)
    source_manifest = dict(manifest, questions_sha256=sha256(source / 'questions.json'))
    if (source / 'manifest.json').exists():
        if json.loads((source / 'manifest.json').read_text()) != source_manifest:
            raise ValueError('Frozen question snapshot changed')
    else:
        atomic_json(source / 'manifest.json', source_manifest)
    for name in (('trajectories',) if screen else ('trajectories', 'features')):
        (source / name).mkdir(exist_ok=True)
    counts = json.loads((source / 'questions.json').read_text())['counts']
    print(f"Selected questions: {counts}; {config['responses']} answers each", flush=True)
    return manifest


def prepare(source, questions):
    import torch
    if (source / 'summary.json').exists():
        return
    prepare_probes(source, questions)
    train = torch.load(source / 'train.pt', weights_only=True, map_location='cpu')
    save_tensor(source / 'normalization.pt', dict(mean=train['x'].float().mean(0),
        scale=train['x'].float().std(0, unbiased=False).clamp_min(1e-5), baseline=float(train['y'].mean())))
    statuses = {s: Counter() for s in ('train', 'val', 'test')}
    for i, q in enumerate(questions):
        raw = json.loads((source / 'trajectories' / f'{i:05d}.json').read_text())
        statuses[q['split']].update(r['verifier_status'] for r in raw['responses'])
    atomic_json(source / 'summary.json', dict(scope='Data preparation only; normalization uses training prefixes only',
        statuses={s: dict(c) for s, c in statuses.items()},
        test_previously_inspected=bool(json.loads((source / 'manifest.json').read_text())['config'].get('source_run'))))


def fit_and_report(out, manifest, questions):
    import numpy as np
    import torch
    torch.set_num_threads(min(8, int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))))
    from fit_lstd import run
    from analyze_lstd import diagnostics, report_text
    source, fitted = out / 'data', out / 'fit'
    config = manifest['config']
    run(source, fitted, config['alphas'], config['lambdas'])
    summary = json.loads((fitted / 'summary.json').read_text())
    data = torch.load(source / 'test.pt', weights_only=True, map_location='cpu')
    predictions = {name: np.load(fitted / f'{method}-predictions.npy') for name, method in
                   (('lstd', summary['selected_lstd']), ('ridge', 'ridge'))}
    length = diagnostics(source, questions, data, predictions['lstd'], predictions['ridge'])
    atomic_json(out / 'length-summary.json', length)
    (out / 'length-report.txt').write_text(report_text(length))
    summary['difficulty'] = supplement(source, questions, data, predictions)
    selection = json.loads((source / 'questions.json').read_text())
    summary['question_selection'] = {k: selection[k] for k in
        ('counts', 'limits', 'split_rule', 'reserved_math500_questions', 'dataset_revisions') if k in selection}
    atomic_json(out / 'summary.json', summary)
    lines = ['MATH-specific fitting; levels 4/5 numeric subset; frozen base actor.',
             ('MATH-500 excluded. Lambda fixed at 0.99; alpha selected on validation only.'
              if config.get('source_run') else 'MATH-500 excluded. Lambda and alpha selected on validation only.'),
             f"Prompt: {config.get('prompt_style', 'completion')}; response budget: {config.get('sampling', {}).get('max_tokens', 2048)}",
             (fitted / 'report.txt').read_text(),
             'Per-level test results: ' + json.dumps(summary['difficulty']),
             'Inspect data/review.jsonl. Critic metrics conditional on verifiable answers. No PPO claim.']
    (out / 'report.txt').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--source-run', type=Path)
    parser.add_argument('--prompt-style', choices=('completion', 'plan'), default='completion')
    parser.add_argument('--budget', type=int, choices=(2048, 4096), default=2048)
    parser.add_argument('--screen', action='store_true', help='Generate only: 100 validation questions, four answers each')
    parser.add_argument('--bounded', action='store_true', help='300/50/100 questions and four answers for planning critics')
    parser.add_argument('--structured', action='store_true')
    parser.add_argument('--multistage', action='store_true')
    parser.add_argument('--stage', choices=('generate', 'extract', 'prepare', 'fit'))
    args = parser.parse_args()
    out = args.out.resolve()
    if args.stage:
        manifest = json.loads((out / 'manifest.json').read_text())
        for file, expected in manifest['files'].items():
            if sha256(file) != expected:
                raise ValueError(f'Locked input changed: {file}')
        source = out / 'data'
        saved = json.loads((source / 'manifest.json').read_text())
        if saved['questions_sha256'] != sha256(source / 'questions.json'):
            raise ValueError('Question snapshot changed')
        questions = json.loads((source / 'questions.json').read_text())['questions']
        if args.stage in ('generate', 'extract'):
            {'generate': generate, 'extract': extract}[args.stage](source, manifest['config'], questions)
        elif args.stage == 'prepare':
            prepare(source, questions)
        else:
            fit_and_report(out, manifest, questions)
        return
    initialize(out, args.source_run, args.prompt_style, args.budget, args.screen, args.bounded, args.structured, args.multistage)
    for stage in (('generate',) if args.screen else ('generate', 'extract', 'prepare', 'fit')):
        started = time.monotonic()
        atomic_json(out / 'status.json', dict(stage=stage, state='running'))
        with (out / f'{stage}.log').open('a') as log:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--out', str(out), '--stage', stage],
                                    stdout=log, stderr=subprocess.STDOUT)
        status = dict(stage=stage, state='complete' if result.returncode == 0 else 'failed',
                      exit_code=result.returncode, seconds=time.monotonic() - started)
        atomic_json(out / 'status.json', status)
        with (out / 'timings.jsonl').open('a') as history:
            history.write(json.dumps(status) + '\n')
        if result.returncode:
            raise RuntimeError(f'{stage} failed; see {out / (stage + ".log")}')
    if not args.screen:
        print((out / 'report.txt').read_text())


if __name__ == '__main__':
    main()
