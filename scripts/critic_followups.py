"""Matched frozen, cumulative and periodically refreshed LSTD-PPO studies."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import tempfile
import numpy as np
import math_comparison as comparison
from math_rl.provenance import snapshot, write_json


def conditions(study, interval):
    if interval < 1:
        raise ValueError('Refresh/reset interval must be positive')
    if study == 'grpo_final':
        return {'grpo': dict(method='grpo', reset=0, refresh=0),
                'lstd-0.99-two_thirds': dict(method='lstd', trace=.99, layer='two_thirds', reset=0, refresh=0)}
    if study == 'layers':
        result = {'ppo': dict(method='ppo', reset=0, refresh=0)}
        for layer in ('two_thirds', 'final'):
            for method, trace in [('ridge', 1.), ('lstd', .95), ('lstd', .99)]:
                name = f'{method}-{trace:g}-{layer}' if method == 'lstd' else f'ridge-{layer}'
                result[name] = dict(method=method, trace=trace, layer=layer, reset=0, refresh=0)
        return result
    if study == 'cumulative':
        return {'frozen_batch': dict(reset=1, refresh=0),
                'frozen_cumulative': dict(reset=0, refresh=0)}
    if study != 'refresh':
        raise ValueError('Unknown study')
    return {'frozen_cumulative': dict(reset=0, refresh=0),
            'frozen_reset': dict(reset=interval, refresh=0),
            'refreshed_reset': dict(reset=interval, refresh=interval)}


def command(out, name, seed, steps, epsilon, trace_lambda, setting):
    method = setting.get('method', 'lstd')
    if method in ('ppo', 'grpo'):
        return comparison.command(out, method, seed, steps, ppo_profile='more_fitting' if method == 'ppo' else 'legacy',
            case_name=f'{name}-seed{seed}')
    trace_lambda = setting.get('trace', trace_lambda)
    return comparison.command(out, method, seed, steps, feature_mode='frozen',
        case_name=f'{name}-seed{seed}') + ['experiment.critic_statistics=cumulative',
        f'actor_rollout_ref.model.critic_feature_layer={setting.get("layer", "final")}',
        f'experiment.cumulative_epsilon={epsilon}', f'experiment.critic_lambda={trace_lambda}',
        f'experiment.statistics_reset_interval={setting["reset"]}',
        f'experiment.encoder_refresh_interval={setting["refresh"]}']


def report(out, manifest):
    questions = json.loads((out / 'data/questions.json').read_text())['questions']
    expected = [(q['id'], q['ground_truth']) for q in questions if q['split'] == 'test']
    runs, scores, methods, reference_base = {}, {}, {}, None
    answer_rows = {}
    for name in manifest['conditions']:
        arrays, costs, accuracies = [], [], []
        for seed in manifest['seeds']:
            case = out / f'{name}-seed{seed}'
            result = json.loads((case / 'result.json').read_text())
            if result['updates'] != manifest['steps'] or result['answers'] != manifest['steps'] * 64:
                raise ValueError('Incomplete update budget')
            rows = comparison.read_answers(case / 'final-test.jsonl')
            base = comparison.read_answers(case / 'base-test.jsonl')
            for batch in (rows, base):
                if [(r['id'], r['ground_truth']) for r in batch] != expected:
                    raise ValueError('Test questions/references differ')
            initial = [r['score'] for r in base]
            if reference_base is None:
                reference_base = initial
            if initial != reference_base:
                raise ValueError('Starting-model scores differ')
            observed_accuracy = float(np.mean([r['score'] for r in rows]))
            if not np.isclose(observed_accuracy, result['final_test']['accuracy'], atol=1e-10, rtol=0):
                raise ValueError('Reported accuracy differs from saved answers')
            answer_rows[(name, seed)] = rows
            result['verifier_statuses'] = dict(Counter(r['verifier_status'] for r in rows))
            result['per_level'] = {str(level): float(np.mean([r['score'] for r in rows if r['level'] == level]))
                for level in (4, 5) if any(r['level'] == level for r in rows)}
            result['truncation_rate_from_answers'] = float(np.mean([r['capped'] for r in rows]))
            result['accounted_gpu_hours'] = result['training_gpu_hours'] + 2 * (
                result['initialization_seconds'] + result['initial_test']['seconds'] + result['final_test']['seconds']) / 3600
            arrays.append([r['score'] for r in rows])
            costs.append(result['training_gpu_hours'])
            accuracies.append(result['final_test']['accuracy'])
            if manifest['conditions'][name].get('method') in ('ppo', 'grpo'):
                runs[case.name] = dict(result=result)
                continue
            metrics = [json.loads(x) for x in (case / 'linear-critic/metrics.jsonl').read_text().splitlines()]
            if [m['step'] for m in metrics] != list(range(1, manifest['steps'] + 1)):
                raise ValueError('Missing critic diagnostics')
            runs[case.name] = dict(result=result, last_critic=metrics[-1],
                fit_seconds=sum(m['fit_seconds'] for m in metrics),
                statistics_resets=sum(m['reset'] for m in metrics))
        scores[name] = np.asarray(arrays)
        methods[name] = dict(accuracy_mean=float(np.mean(accuracies)),
            accuracy_seed_sd=float(np.std(accuracies, ddof=1)) if len(accuracies)>1 else None,
            gpu_hours_mean=float(np.mean(costs)),
            gain_over_base_pp=100 * (float(np.mean(accuracies)) - float(np.mean(reference_base))),
            accounted_gpu_hours_mean=float(np.mean([runs[f'{name}-seed{s}']['result']['accounted_gpu_hours'] for s in manifest['seeds']])),
            accuracy_by_seed=dict(zip(map(str,manifest['seeds']),accuracies)))
    paired = {}
    names = list(scores)
    for name, reference in [(names[i], names[j]) for i in range(len(names)) for j in range(i)]:
        delta = scores[name] - scores[reference]
        rng = np.random.default_rng(912)
        draws = [float(delta[np.ix_(rng.integers(len(delta),size=len(delta)),
                                     rng.integers(delta.shape[1],size=delta.shape[1]))].mean()*100)
                 for _ in range(2000)]
        paired[f'{name}_minus_{reference}'] = dict(mean_pp=float(delta.mean()*100),
            seed_differences_pp=dict(zip(map(str, manifest['seeds']), (delta.mean(1)*100).tolist())),
            training_gpu_hours_ratio=methods[name]['gpu_hours_mean']/methods[reference]['gpu_hours_mean'],
            descriptive_95_interval_pp=np.quantile(draws,[.025,.975]).tolist())
    summary = dict(methods=methods, runs=runs, paired=paired,
        scope='Inspected MATH split; uncorrected historical-policy critic fitting. Actor updates use fresh data only. '
              f'Seed/question bootstrap is exploratory; {len(manifest["seeds"])} training seeds; no superiority gate.')
    if manifest['study'] == 'grpo_final':
        summary['primary_comparison'] = 'lstd-0.99-two_thirds_minus_grpo'
        summary['cost_scope'] = 'Training includes frozen feature extraction, fitting, validation and checkpoints; accounted cost additionally includes initialization and base/final test evaluation, not environment setup or queue time.'
        summary['audit'] = write_final_audit(out, manifest, answer_rows, questions)
    write_json(out / 'summary.json',summary)
    lines = [f"Frozen encoder study: {manifest['study']}; raw sums, epsilon={manifest['epsilon']}",
             'No per-update normalization; fixed coordinates and persistent question folds.',
             'method                  accuracy mean    GPU hours mean']
    lines += [f"{name:25} {r['accuracy_mean']:.4f}           {r['gpu_hours_mean']:.3f}" for name,r in methods.items()]
    lines += [f'{key}: {value}' for key,value in paired.items()]
    for name, value in methods.items():
        lines.append(f"{name} accuracy by seed: {value['accuracy_by_seed']}; SD: {value['accuracy_seed_sd']}; accounted GPUh: {value['accounted_gpu_hours_mean']:.3f}")
    if manifest['study'] == 'grpo_final':
        lines.append('Locked LSTD configuration; fresh training seeds, same inspected questions. Review audit.jsonl before any final accuracy claim. An interval spanning zero is inconclusive, not equivalence.')
    lines.append(summary['scope'])
    (out / 'report.txt').write_text('\n'.join(lines)+'\n')



def write_final_audit(out, manifest, rows, questions):
    """Keep all test answers available; prioritize disagreements without changing scores."""
    prompts = {q['id']: q['question'] for q in questions}
    records = []
    for seed in manifest['seeds']:
        left = rows[('grpo', seed)]
        right = rows[('lstd-0.99-two_thirds', seed)]
        for g, l in zip(left, right, strict=True):
            reasons = []
            if g['score'] != l['score']:
                reasons.append('method_disagreement')
            if any(r['verifier_status'] not in ('correct', 'incorrect') for r in (g, l)):
                reasons.append('verifier_failure')
            if any(r['capped'] for r in (g, l)):
                reasons.append('capped')
            records.append(dict(seed=seed, id=g['id'], question=prompts[g['id']],
                reasons=reasons, grpo=g, lstd=l))
    # Preserve every answer, including agreement cases, for an unbiased review sample.
    records.sort(key=lambda r: not bool(r['reasons']))
    with (out / 'audit.jsonl').open('w') as handle:
        for record in records:
            handle.write(json.dumps(record) + '\n')
    return dict(pairs=len(records), flagged=sum(bool(r['reasons']) for r in records),
        review_complete=False, note='Flags prioritize review; they are not proof of scoring errors. Review a random sample of agreements as well.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--study',choices=['cumulative','refresh','layers','grpo_final'],default='cumulative')
    p.add_argument('--steps',type=int,default=60)
    p.add_argument('--seeds',type=int,nargs='+',default=None)
    p.add_argument('--epsilon',type=float,default=.01)
    p.add_argument('--trace-lambda',type=float,default=.99)
    p.add_argument('--refresh-interval',type=int,default=10)
    args=p.parse_args()
    if args.seeds is None:
        args.seeds = [101,102,103,104,105] if args.study == 'grpo_final' else [42,43,44]
    if args.study == 'grpo_final' and (args.epsilon != .01 or args.trace_lambda != .99):
        p.error('Final comparison locks epsilon=.01 and LSTD lambda=.99')
    if (args.steps<1 or not np.isfinite(args.epsilon) or args.epsilon<=0
            or not 0<=args.trace_lambda<=1 or len(set(args.seeds))!=len(args.seeds)):
        p.error('Invalid budgets, regularization, trace or seeds')
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    state=dict(state='preparing');write_json(out/'status.json',state)
    try:
        data=comparison.prepare(args.source,out)
        manifest=dict(protocol='critic-followups-v1',study=args.study,conditions=conditions(args.study,args.refresh_interval),
            steps=args.steps,seeds=args.seeds,epsilon=args.epsilon,**{'lambda':args.trace_lambda},data=data,
            feature='Frozen Qwen; condition-specific layer (default final RMSNorm); raw coordinates plus penalized intercept',
            fairness='All heads start with zero statistics; same updates, answers per update, seeds, folds and epsilon; no offline weights loaded. PPO uses locked more_fitting profile in layers study.',
            regularization='Raw A,b sums; epsilon*I added once at solve, never divided by count',
            primary_comparison='lstd-0.99-two_thirds_minus_grpo' if args.study == 'grpo_final' else None,
            final_protocol='Same rollout budget; no tuning or checkpoint selection; five fresh seeds by default; same inspected questions, not independent confirmation.',
            refresh='Copy current actor at updates 1+interval, 1+2*interval; reset A,b',
            provenance=snapshot(comparison.ROOT))
        write_json(out/'manifest.json',manifest)
        for si,seed in enumerate(args.seeds):
            names=list(manifest['conditions']);offset=si%len(names);names=names[offset:]+names[:offset]
            for name in names:
                state.update(state='training',current=f'{name}-seed{seed}');write_json(out/'status.json',state)
                cmd=command(out,name,seed,args.steps,args.epsilon,args.trace_lambda,manifest['conditions'][name])
                env=dict(os.environ,RAY_ADDRESS='local',MATH_RL_SEED=str(seed),RAY_TMPDIR=tempfile.mkdtemp(prefix='mr-',dir='/tmp'))
                with (out/f'{name}-seed{seed}.log').open('x') as log:
                    subprocess.run(cmd,cwd=comparison.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        report(out,manifest);state.update(state='complete')
    except BaseException as exc:
        state.update(state='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        write_json(out/'status.json',state)

if __name__=='__main__':
    main()
