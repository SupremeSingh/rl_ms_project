"""Matched frozen, cumulative and periodically refreshed LSTD-PPO studies."""
import argparse
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
    if study == 'cumulative':
        return {'frozen_batch': dict(reset=1, refresh=0),
                'frozen_cumulative': dict(reset=0, refresh=0)}
    if study != 'refresh':
        raise ValueError('Unknown study')
    return {'frozen_cumulative': dict(reset=0, refresh=0),
            'frozen_reset': dict(reset=interval, refresh=0),
            'refreshed_reset': dict(reset=interval, refresh=interval)}


def command(out, name, seed, steps, epsilon, trace_lambda, setting):
    return comparison.command(out, 'lstd', seed, steps, feature_mode='frozen',
        case_name=f'{name}-seed{seed}') + ['experiment.critic_statistics=cumulative',
        f'experiment.cumulative_epsilon={epsilon}', f'experiment.critic_lambda={trace_lambda}',
        f'experiment.statistics_reset_interval={setting["reset"]}',
        f'experiment.encoder_refresh_interval={setting["refresh"]}']


def report(out, manifest):
    questions = json.loads((out / 'data/questions.json').read_text())['questions']
    expected = [(q['id'], q['ground_truth']) for q in questions if q['split'] == 'test']
    runs, scores, methods, reference_base = {}, {}, {}, None
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
            arrays.append([r['score'] for r in rows])
            costs.append(result['training_gpu_hours'])
            accuracies.append(result['final_test']['accuracy'])
            metrics = [json.loads(x) for x in (case / 'linear-critic/metrics.jsonl').read_text().splitlines()]
            if len(metrics) != manifest['steps']:
                raise ValueError('Missing critic diagnostics')
            runs[case.name] = dict(result=result, last_critic=metrics[-1],
                fit_seconds=sum(m['fit_seconds'] for m in metrics),
                statistics_resets=sum(m['reset'] for m in metrics))
        scores[name] = np.asarray(arrays)
        methods[name] = dict(accuracy_mean=float(np.mean(accuracies)),
            accuracy_seed_sd=float(np.std(accuracies, ddof=1)) if len(accuracies)>1 else None,
            gpu_hours_mean=float(np.mean(costs)), accuracy_by_seed=dict(zip(map(str,manifest['seeds']),accuracies)))
    paired = {}
    names = list(scores)
    for name, reference in [(names[i], names[j]) for i in range(len(names)) for j in range(i)]:
        delta = scores[name] - scores[reference]
        rng = np.random.default_rng(912)
        draws = [float(delta[np.ix_(rng.integers(len(delta),size=len(delta)),
                                     rng.integers(delta.shape[1],size=delta.shape[1]))].mean()*100)
                 for _ in range(2000)]
        paired[f'{name}_minus_{reference}'] = dict(mean_pp=float(delta.mean()*100),
            descriptive_95_interval_pp=np.quantile(draws,[.025,.975]).tolist())
    summary = dict(methods=methods, runs=runs, paired=paired,
        scope='Inspected MATH split; uncorrected historical-policy critic fitting. Actor updates use fresh data only. '
              'Seed/question bootstrap is exploratory; one seed cannot measure training variability.')
    write_json(out / 'summary.json',summary)
    lines = [f"LSTD({manifest['lambda']}) frozen encoder follow-up; raw sums, epsilon={manifest['epsilon']}",
             'No per-update normalization; fixed coordinates and persistent question folds.',
             'method                  accuracy mean    GPU hours mean']
    lines += [f"{name:25} {r['accuracy_mean']:.4f}           {r['gpu_hours_mean']:.3f}" for name,r in methods.items()]
    lines += [f'{key}: {value}' for key,value in paired.items()]
    lines.append(summary['scope'])
    (out / 'report.txt').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--study',choices=['cumulative','refresh'],default='cumulative')
    p.add_argument('--steps',type=int,default=60)
    p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    p.add_argument('--epsilon',type=float,default=.01)
    p.add_argument('--trace-lambda',type=float,default=.99)
    p.add_argument('--refresh-interval',type=int,default=10)
    args=p.parse_args()
    if (args.steps<1 or not np.isfinite(args.epsilon) or args.epsilon<=0
            or not 0<=args.trace_lambda<=1 or len(set(args.seeds))!=len(args.seeds)):
        p.error('Invalid budgets, regularization, trace or seeds')
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    state=dict(state='preparing');write_json(out/'status.json',state)
    try:
        data=comparison.prepare(args.source,out)
        manifest=dict(protocol='critic-followups-v1',study=args.study,conditions=conditions(args.study,args.refresh_interval),
            steps=args.steps,seeds=args.seeds,epsilon=args.epsilon,**{'lambda':args.trace_lambda},data=data,
            feature='Frozen Qwen final normalized hidden state; raw coordinates plus penalized intercept',
            regularization='Raw A,b sums; epsilon*I added once at solve, never divided by count',
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
