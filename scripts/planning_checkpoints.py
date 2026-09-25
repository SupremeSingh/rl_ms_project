"""Causal prefix diagnostics and escaped, standalone response viewer for planning critics."""
from collections import defaultdict
from html import escape
import json
import re

import numpy as np
import torch

from fit_lstd import read_shard
from frozen_critics import atomic_json
from math_rl.lstd import predict

POSITIONS = (0, 32, 64, 128, 256)
METHODS = ('lstd-0.99', 'ridge')
BOUNDARY_VERSION = 'explicit-headings-v2'


def headings(text, title):
    """Recognize explicit Markdown/plain headings, including inline content; ignore code."""
    fences = [(m.start(), m.end()) for m in re.finditer(r'```.*?(?:```|\Z)', text, re.S)]
    pattern = r'(?im)^[ \t]*(?:#{1,6}[ \t]+)?(?:\*\*)?' + title + r'(?:\*\*)?:[ \t]*(?:\*\*)?[ \t]*(?:\r?\n)?'
    return [m for m in re.finditer(pattern, text)
            if not any(lo <= m.start() < hi for lo, hi in fences)]


def format_audit(text):
    known, actions = headings(text, 'What we know'), headings(text, 'What we will do')
    solution = headings(text, r'(?:Step[- ]by[- ]step[ \t]+)?Solution')
    final = headings(text, 'Final answer')
    ordered = all(len(group) == 1 for group in (known, actions, solution, final))
    ordered = ordered and known[0].start() < actions[0].start() < solution[0].start() < final[0].start()
    populated = ordered and all(text[a.end():b.start()].strip() for a, b in
                               ((known[0], actions[0]), (actions[0], solution[0]), (solution[0], final[0])))
    populated = populated and bool(text[final[0].end():].strip())
    return dict(ordered_sections=bool(populated), solution_headings=len(solution),
                answer_before_solution=bool(solution and r'\boxed' in text[:solution[0].start()]),
                boundary_version=BOUNDARY_VERSION)


def plan_boundary(tokenizer, ids):
    """First complete explicit heading; map using decoded prefixes, never re-tokenize."""
    text = tokenizer.decode(ids, skip_special_tokens=True)
    matches = headings(text, r'(?:Step[- ]by[- ]step[ \t]+)?Solution')
    if len(matches) != 1 or not text[:matches[0].start()].strip():
        return None
    end = matches[0].end()
    for length in range(1, len(ids)):
        prefix = tokenizer.decode(ids[:length], skip_special_tokens=True)
        if len(prefix) >= end and prefix[:end] == text[:end]:
            return length
    return None


def metrics(rows):
    y, p = np.array([r['y'] for r in rows]), np.array([r['p'] for r in rows])
    return dict(examples=len(rows), questions=len({r['question'] for r in rows}),
        brier=float(np.mean((p - y) ** 2)), accuracy=float(np.mean((p >= .5) == y)),
        outcome_rate=float(y.mean()), mean_prediction=float(p.mean()),
        out_of_range_rate=float(np.mean((p < 0) | (p > 1))),
        calibration=[dict(count=int(mask.sum()), mean_prediction=float(p[mask].mean()),
                          success_rate=float(y[mask].mean()))
                     for low in np.arange(0, 1, .2)
                     if (mask := ((np.clip(p, 0, .999999) >= low) &
                                  (np.clip(p, 0, .999999) < low + .2))).any()])


def paired(left, right):
    """Question-weighted Brier difference, left minus right."""
    def errors(rows):
        groups = defaultdict(list)
        for row in rows:
            groups[row['question']].append((row['p'] - row['y']) ** 2)
        return {key: np.mean(values) for key, values in groups.items()}
    a, b = errors(left), errors(right)
    keys = sorted(a.keys() & b.keys())
    if not keys:
        return None
    differences = np.array([a[key] - b[key] for key in keys])
    rng = np.random.default_rng(42)
    interval = np.quantile([rng.choice(differences, len(keys), replace=True).mean()
                           for _ in range(2000)], [.025, .975]).tolist()
    return dict(questions=len(keys), difference=float(differences.mean()), interval_95=interval)


def render_viewer(path, examples):
    sections = []
    for example in examples:
        columns = []
        for style in ('completion', 'plan'):
            answer = example[style]
            columns.append('<article><h3>' + style + '</h3><p>' + escape(answer['status']) +
                ' · plan boundary: ' + escape(str(answer['plan_boundary'])) +
                '</p><pre>' + escape(answer['response']) + '</pre><h4>Prefix predictions</h4><pre>' +
                escape(json.dumps(answer['predictions'], indent=2)) + '</pre><h4>Audit</h4><pre>' +
                escape(json.dumps(dict(format=answer.get('format_audit'), verifier=answer.get('verifier_audit')), indent=2)) +
                '</pre></article>')
        sections.append('<details><summary>' + escape(example['id']) + ' · sample ' +
            str(example['sample']) + ' · ' + escape(example['selection']) + '</summary><p>' +
            escape(example['question']) + '</p><p>Reference: ' + escape(example['reference']) +
            '</p><div class="columns">' + ''.join(columns) + '</div></details>')
    path.write_text('<!doctype html><meta charset="utf-8"><title>Planning critic review</title>'
        '<style>body{font:16px system-ui;max-width:1400px;margin:30px auto;padding:20px}'
        '.columns{display:grid;grid-template-columns:1fr 1fr;gap:25px}pre{white-space:pre-wrap;overflow-wrap:anywhere}'
        'details{border-top:1px solid #ccc;padding:15px}summary{cursor:pointer}article{min-width:0}'
        '@media(max-width:700px){.columns{grid-template-columns:1fr}}</style>'
        '<h1>Planning critic review</h1><p>Frozen base model. Sample indices do not identify paired stochastic answers. '
        'Prefix 0 means before any answer tokens. Predictions are raw linear values, not clipped probabilities. '
        'A detected heading is not proof of a sensible plan. Verify the boundary and answer manually. '
        'Excluded responses remain visible without predictions.</p>' + ''.join(sections))


def evaluate(out, tokenizer=None):
    if tokenizer is None:
        from transformers import AutoTokenizer
        from frozen_critics import ROOT
        tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/qwen-math', local_files_only=True)
    groups, records, complete = defaultdict(list), {}, {}
    for style in ('completion', 'plan'):
        run = out / f'{style}-2048'
        questions = json.loads((run / 'data/questions.json').read_text())['questions']
        heads = {m: torch.load(run / f'fit/{m}.pt', weights_only=True, map_location='cpu') for m in METHODS}
        constant = float(torch.load(run / 'data/normalization.pt', weights_only=True)['baseline'])
        rows = {}
        for index, q in enumerate(questions):
            if q['split'] != 'test':
                continue
            shard, raw, _ = read_shard(run / 'data', index, q)
            retained = {int(v): j for j, v in enumerate(shard['response_indices'])}
            for sample, answer in enumerate(raw['responses']):
                boundary = plan_boundary(tokenizer, answer['token_ids']) if style == 'plan' else None
                record = dict(response=answer['response'], status=answer['verifier_status'],
                    plan_boundary=boundary, format_audit=format_audit(answer['response']),
                    verifier_audit=answer.get('audit'), predictions={}, question=q['question'], reference=q['ground_truth'])
                rows[(q['id'], sample)] = record
                if sample not in retained:
                    continue
                j = retained[sample]
                lo, hi = shard['offsets'][j:j + 2].tolist()
                # Exclude terminal feature: every prediction precedes a next action.
                x = shard['features'][lo:hi - 1]
                values = {m: predict(h['weights'], {'x': x}, h['mean'], h['scale']) for m, h in heads.items()}
                values['constant'] = np.full(len(x), constant)
                complete[(style, q['id'], sample)] = dict(values=values, y=answer['score'])
                points = {str(p): p for p in POSITIONS if p < len(x)}
                if boundary is not None and boundary < len(x):
                    points['post_plan'] = boundary
                for label, position in points.items():
                    record['predictions'][label] = {m: float(v[position]) for m, v in values.items()}
                    for m, v in values.items():
                        groups[(style, label, m)].append(dict(question=q['id'], y=answer['score'], p=float(v[position])))
        records[style] = rows
    if records['completion'].keys() != records['plan'].keys():
        raise ValueError('Response question/sample IDs differ')
    # Compare each identified plan length with every eligible ordinary answer for that question.
    for (qid, sample), record in records['plan'].items():
        length = record['plan_boundary']
        target = complete.get(('plan', qid, sample))
        if length is None or target is None:
            continue
        controls = [complete[('completion', qid, s)] for s in range(4)
                    if ('completion', qid, s) in complete and
                    len(complete[('completion', qid, s)]['values']['ridge']) > length]
        if not controls:
            continue
        for method in (*METHODS, 'constant'):
            groups[('plan', 'matched_plan_length', method)].append(
                dict(question=qid, y=target['y'], p=float(target['values'][method][length])))
            for control in controls:
                groups[('completion', 'matched_plan_length', method)].append(
                    dict(question=qid, y=control['y'], p=float(control['values'][method][length])))
    comparisons = {}
    for label in (*map(str, POSITIONS), 'matched_plan_length'):
        for method in (*METHODS, 'constant'):
            comparisons[f'{label}/{method}'] = paired(groups[('plan', label, method)], groups[('completion', label, method)])
    report = dict(metrics={'/'.join(key): metrics(rows) for key, rows in groups.items() if rows},
        paired_plan_minus_completion=comparisons,
        paired_critic_minus_constant={f'{style}/{label}/{method}': paired(
            rows, groups[(style, label, 'constant')])
            for (style, label, method), rows in list(groups.items()) if method in METHODS and rows},
        plan_heading_detected=sum(r['plan_boundary'] is not None for r in records['plan'].values()),
        total_plan_answers=len(records['plan']),
        boundary_version=BOUNDARY_VERSION,
        ordered_plan_sections=sum(r['format_audit']['ordered_sections'] for r in records['plan'].values()),
        early_boxed_answers=sum(r['format_audit']['answer_before_solution'] for r in records['plan'].values()),
        scope='Exploratory inspected test questions. Retained answers only; prefix must precede termination. '
              'Coverage and outcome rates vary with prompt and position. Matched lengths condition on observed plan '
              'length and ordinary-answer survival, not a causal planning effect. Confidence intervals cluster '
              'by question and are unadjusted. Plan headings require manual compliance review.')
    atomic_json(out / 'checkpoints.json', report)
    rng = np.random.default_rng(42)
    keys = sorted(records['plan'])
    random_keys = {keys[i] for i in rng.choice(len(keys), min(12, len(keys)), replace=False)}
    def error(key):
        candidates = [abs(r['predictions']['0']['ridge'] - complete[(style, *key)]['y'])
                      for style in records if '0' in (r := records[style][key])['predictions']]
        return max(candidates, default=-1)
    worst_keys = set(sorted(keys, key=error, reverse=True)[:12])
    examples = []
    for key in sorted(random_keys | worst_keys):
        row = records['plan'][key]
        examples.append(dict(id=key[0], sample=key[1], question=row['question'], reference=row['reference'],
            selection='random' if key in random_keys else 'largest question-only ridge error',
            **{style: records[style][key] for style in records}))
    atomic_json(out / 'viewer-examples.json', examples)
    render_viewer(out / 'responses.html', examples)
    lines = ['\nPrefix diagnostics: plan minus completion Brier (negative favors planning)']
    lines.append(f"Format audit: {report['ordered_plan_sections']}/{report['total_plan_answers']} answers with ordered, nonempty sections; "
                 f"{report['plan_heading_detected']} detected boundaries; {report['early_boxed_answers']} early boxed answers. "
                 f"Boundary rule: {BOUNDARY_VERSION}. These are syntax checks, not reasoning checks.")
    for name, comparison in comparisons.items():
        if comparison:
            lines.append(f"{name}: {comparison['difference']:+.5f}; 95% interval {comparison['interval_95']}; questions={comparison['questions']}")
    lines.append(report['scope'])
    lines.append('Inspect checkpoints.json and responses.html; all full responses remain in each data/trajectories directory.')
    with (out / 'report.txt').open('a') as stream:
        stream.write('\n'.join(lines) + '\n')
