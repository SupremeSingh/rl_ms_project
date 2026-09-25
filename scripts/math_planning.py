"""Matched frozen-base prompt experiment; no PPO updates or test-based selection."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from frozen_critics import atomic_json

ROOT = Path(__file__).resolve().parents[1]


def conditions(long_budget=False):
    return [(style, budget) for budget in ([2048, 4096] if long_budget else [2048])
            for style in ('completion', 'plan')]


def summarize_screen(out):
    """Keep failed extractions in the denominator; pair at question, not answer."""
    from collections import Counter
    import numpy as np
    results, records, review = {}, {}, []
    for style in ('completion', 'plan'):
        run = out / f'{style}-2048'
        questions = json.loads((run / 'data/questions.json').read_text())['questions']
        rows, statuses, lengths, capped = {}, Counter(), [], 0
        for index, q in enumerate(questions):
            if q['split'] != 'val':
                raise ValueError('Screen must use validation only')
            shard = json.loads((run / 'data/trajectories' / f'{index:05d}.json').read_text())
            answers = shard['responses']
            if shard['question_id'] != q['id'] or len(answers) != 4 or q['id'] in rows:
                raise ValueError('Invalid screening trajectory')
            statuses.update(r['verifier_status'] for r in answers)
            lengths.extend(len(r['token_ids']) for r in answers)
            capped += sum(r['finish_reason'] == 'length' for r in answers)
            rows[q['id']] = dict(question=q['question'], ground_truth=q['ground_truth'], level=q['level'],
                accuracy=sum(r['verifier_status'] == 'correct' for r in answers) / 4,
                responses=answers)
        records[style] = rows
        results[style] = dict(questions=len(rows), answers=len(lengths),
            answer_accuracy=statuses['correct'] / len(lengths),
            truncation_rate=capped / len(lengths), mean_response_tokens=float(np.mean(lengths)),
            verifier_statuses=dict(statuses),
            per_level={str(level): float(np.mean([r['accuracy'] for r in rows.values() if r['level'] == level]))
                       for level in (4, 5)})
    if records['completion'].keys() != records['plan'].keys():
        raise ValueError('Screening question IDs differ')
    differences = []
    for key, baseline in records['completion'].items():
        planned = records['plan'][key]
        if (baseline['question'], baseline['ground_truth'], baseline['level']) != (
                planned['question'], planned['ground_truth'], planned['level']):
            raise ValueError('Screening questions differ')
        difference = planned['accuracy'] - baseline['accuracy']
        differences.append(difference)
        review.append(dict(id=key, accuracy_change=difference, **{
            style: records[style][key] for style in records}))
    # Stratified question bootstrap preserves the balanced level mix.
    rng = np.random.default_rng(42)
    groups = [[planned['accuracy'] - records['completion'][key]['accuracy']
               for key, planned in records['plan'].items() if planned['level'] == level]
              for level in (4, 5)]
    interval = np.quantile([np.mean([rng.choice(g, len(g), replace=True).mean() for g in groups])
                           for _ in range(2000)], [.025, .975]).tolist()
    report = dict(conditions=results,
        plan_minus_completion=dict(accuracy_change_percentage_points=100 * float(np.mean(differences)),
            interval_95_percentage_points=[100 * v for v in interval]),
        scope='Validation screening only; frozen base actor; no feature extraction or critic fitting. '
              'All attempts count, including verifier failures. Interval bootstraps questions within levels. '
              'Inspect verifier statuses and review.jsonl; no automatic winner or claim about critic quality. '
              'Plan compliance requires manual review; equal sampling seeds do not pair individual answers.')
    atomic_json(out / 'summary.json', report)
    # Put changed questions first, but retain all answers for compliance/error review.
    review.sort(key=lambda row: (-abs(row['accuracy_change']), row['id']))
    with (out / 'review.jsonl').open('w') as stream:
        for row in review:
            stream.write(json.dumps(row) + '\n')
    lines = [report['scope'], 'prompt       answer accuracy   truncation   mean tokens']
    for style, result in results.items():
        lines.append(f"{style:12} {result['answer_accuracy']:.3f}             {result['truncation_rate']:.3f}        {result['mean_response_tokens']:.1f}")
    lines.append('Paired planning effect: ' + json.dumps(report['plan_minus_completion']))
    (out / 'report.txt').write_text('\n'.join(lines) + '\n')


def summarize(out, names):
    import numpy as np
    import torch
    results, per_question = {}, {}
    for name in names:
        run = out / name
        fit = json.loads((run / 'summary.json').read_text())
        questions = json.loads((run / 'data/questions.json').read_text())['questions']
        probe = torch.load(run / 'data/test.pt', weights_only=True, map_location='cpu')
        predictions = {method: np.load(run / f'fit/{method}-predictions.npy')
                       for method in (fit['selected_lstd'], 'ridge')}
        metrics = {}
        total = correct = retained = capped = tokens = 0
        for index, q in enumerate(questions):
            if q['split'] != 'test':
                continue
            responses = json.loads((run / 'data/trajectories' / f'{index:05d}.json').read_text())['responses']
            wins = sum(r['verifier_status'] == 'correct' for r in responses)
            total += len(responses)
            correct += wins
            retained += sum(r['verifier_status'] in ('correct', 'incorrect') for r in responses)
            capped += sum(r['finish_reason'] == 'length' for r in responses)
            tokens += sum(len(r['token_ids']) for r in responses)
            row = {'answer_accuracy': wins / len(responses)}
            mask = probe['question'].numpy() == q['source_index']
            for method in (fit['selected_lstd'], 'ridge'):
                prediction = predictions[method]
                row[method] = float(np.mean((prediction[mask] - probe['y'].numpy()[mask]) ** 2)) if mask.any() else None
            metrics[q['id']] = row
        per_question[name] = metrics
        results[name] = dict(answer_accuracy=correct / total, correct=correct, answers=total,
            retained=retained, truncation_rate=capped / total, mean_response_tokens=tokens / total,
            critics=fit['methods'], paired_lstd_minus_ridge=fit['paired_question_brier_selected_lstd_minus_ridge'],
            paired_interval=fit['paired_95pct_interval'])
    paired = {}
    for name in names:
        if not name.startswith('plan-'):
            continue
        base = 'completion-' + name.split('-', 1)[1]
        if per_question[name].keys() != per_question[base].keys():
            raise ValueError('Condition question IDs differ')
        paired[name] = {}
        for metric in ('answer_accuracy', 'lstd-0.99', 'ridge'):
            differences = [values[metric] - per_question[base][key][metric]
                for key, values in per_question[name].items()
                if values[metric] is not None and per_question[base][key][metric] is not None]
            if not differences:
                raise ValueError('No shared evaluable questions')
            rng = np.random.default_rng(42)
            interval = np.quantile([rng.choice(differences, len(differences), replace=True).mean()
                                   for _ in range(2000)], [.025, .975]).tolist()
            paired[name][metric] = dict(plan_minus_completion=float(np.mean(differences)),
                                       questions=len(differences), interval_95=interval)
    report = dict(conditions=results, paired=paired,
        scope='Frozen base actor, same inspected MATH questions and generation seeds; no policy update. '
              'Answer accuracy counts all attempts; critic errors condition on retained answers. '
              'Different prompts produce different trajectories and exclusions. '
              'Intervals cluster by question, exploratory and unadjusted. '
              'Random-prefix metrics do not isolate post-plan states or prove improved initial values.')
    atomic_json(out / 'summary.json', report)
    lines = [report['scope'], 'condition             answer accuracy   truncation   mean tokens']
    for name, result in results.items():
        lines.append(f"{name:22} {result['answer_accuracy']:.3f}             {result['truncation_rate']:.3f}        {result['mean_response_tokens']:.1f}")
        lines.append((out / name / 'fit/report.txt').read_text())
    lines.append('Paired planning effects: ' + json.dumps(paired))
    (out / 'report.txt').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_run', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--long-budget', action='store_true', help='Also run both prompts with 4096 response tokens')
    parser.add_argument('--screen', action='store_true', help='Quick validation-only generation comparison; no critic fitting')
    parser.add_argument('--bounded', action='store_true', help='Smaller critic experiment with prefix diagnostics and HTML viewer')
    parser.add_argument('--structured', action='store_true', help='Bounded known-facts/actions prompt and audited conclusion rule')
    parser.add_argument('--audit-source', type=Path, help='Audit a previous planning run into a separate subdirectory')
    args = parser.parse_args()
    if args.structured:
        args.bounded = True
    if args.audit_source and not args.structured:
        parser.error('--audit-source requires --structured')
    if args.screen and args.long_budget:
        parser.error('--screen uses a fixed 2048-token budget; omit --long-budget')
    if args.bounded and (args.screen or args.long_budget):
        parser.error('--bounded cannot be combined with --screen or --long-budget')
    out = args.out.resolve()
    source = args.source_run.resolve()
    if out == source or source in out.parents:
        raise ValueError('Keep new outputs outside the original run')
    out.mkdir(parents=True, exist_ok=True)
    config = dict(source=str(source), conditions=conditions(args.long_budget))
    if args.screen:
        config['screen'] = True
    if args.bounded:
        config['bounded'] = True
    if args.structured:
        config['structured'] = True
        config['audit_source'] = str(args.audit_source.resolve()) if args.audit_source else None
    manifest = out / 'experiment.json'
    if manifest.exists() and json.loads(manifest.read_text()) != json.loads(json.dumps(config)):
        raise ValueError('Resume settings changed')
    atomic_json(manifest, config)
    if args.audit_source:
        audit_out = out / 'source-audit'
        if not (audit_out / 'summary.json').exists():
            atomic_json(out / 'status.json', dict(stage='source-audit', state='running'))
            with (out / 'source-audit.log').open('a') as log:
                result = subprocess.run([sys.executable, str(ROOT / 'scripts/audit_planning.py'),
                    str(args.audit_source.resolve()), '--out', str(audit_out)], stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                atomic_json(out / 'status.json', dict(stage='source-audit', state='failed'))
                raise RuntimeError('Source audit failed; inspect source-audit.log')
    for style, budget in conditions(args.long_budget):
        name = f'{style}-{budget}'
        atomic_json(out / 'status.json', dict(condition=name, state='running'))
        start = time.monotonic()
        with (out / f'{name}.log').open('a') as log:
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/fit_math_critics.py'),
                '--source-run', str(source), '--out', str(out / name),
                '--prompt-style', style, '--budget', str(budget)] +
                (['--screen'] if args.screen else []) + (['--bounded'] if args.bounded else []) +
                (['--structured'] if args.structured else []),
                stdout=log, stderr=subprocess.STDOUT)
        status = dict(condition=name, state='complete' if result.returncode == 0 else 'failed',
                      seconds=time.monotonic() - start, exit_code=result.returncode)
        atomic_json(out / 'status.json', status)
        with (out / 'timings.jsonl').open('a') as log:
            log.write(json.dumps(status) + '\n')
        if result.returncode:
            raise RuntimeError(f'{name} failed; see {out / name} stage logs')
    if args.screen:
        summarize_screen(out)
    else:
        summarize(out, [f'{style}-{budget}' for style, budget in conditions(args.long_budget)])
        if args.bounded:
            from planning_checkpoints import evaluate
            atomic_json(out / 'status.json', dict(stage='checkpoints', state='running'))
            try:
                evaluate(out)
            except Exception:
                atomic_json(out / 'status.json', dict(stage='checkpoints', state='failed'))
                raise
    atomic_json(out / 'status.json', dict(state='complete'))
    print((out / 'report.txt').read_text())


if __name__ == '__main__':
    main()
