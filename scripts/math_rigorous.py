"""Validation-only PPO calibration, then locked, no-buffer multi-seed comparison."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import math_comparison as comparison
from math_rl.ppo_baseline import PPO_PROFILES, audit_health, select_profile
from math_rl.provenance import snapshot, write_json


def launch(out, method, seed, steps, profile='legacy', calibration=False, name=None):
    name = name or f'{method}-seed{seed}'
    cmd = comparison.command(out, method, seed, steps, feature_mode='actor', replay_capacity=0,
        ppo_profile=profile, evaluation_split='val' if calibration else 'test', case_name=name)
    env = dict(os.environ, RAY_ADDRESS='local', MATH_RL_SEED=str(seed),
               RAY_TMPDIR=f'/tmp/mrl-{os.environ.get("SLURM_JOB_ID", os.getpid())}-{name}')
    Path(env['RAY_TMPDIR']).mkdir(exist_ok=True)
    started = time.perf_counter()
    with (out / f'{name}.log').open('x') as log:
        subprocess.run(cmd, cwd=comparison.ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = json.loads((out / name / 'result.json').read_text())
    if result['updates'] != steps or result['answers'] != steps * 64:
        raise ValueError('Incomplete run budget')
    result['process_seconds'] = time.perf_counter() - started
    return result


def aggregate(out, seeds):
    """Paired seed/question resampling; exploratory on already inspected test data."""
    scores, stats = {}, {}
    for method in comparison.METHODS:
        arrays, costs, gains = [], [], []
        for seed in seeds:
            case = out / f'{method}-seed{seed}'
            rows = comparison.read_answers(case / 'final-test.jsonl')
            result = json.loads((case / 'result.json').read_text())
            arrays.append([r['score'] for r in rows])
            costs.append(result['training_gpu_hours'])
            gains.append(result['final_test']['accuracy'] - result['initial_test']['accuracy'])
        scores[method] = np.asarray(arrays)
        accuracies = scores[method].mean(1)
        stats[method] = dict(accuracy_mean=float(accuracies.mean()), accuracy_seed_sd=float(accuracies.std(ddof=1)),
            accuracy_by_seed=dict(zip(map(str, seeds), accuracies.tolist())),
            gain_pp_mean=float(np.mean(gains) * 100), gpu_hours_mean=float(np.mean(costs)))
    intervals = {}
    rng = np.random.default_rng(912)
    for method, reference in [('lstd', 'ppo'), ('ridge', 'ppo'), ('lstd', 'ridge'), ('lstd', 'grpo')]:
        delta = scores[method] - scores[reference]
        draws = []
        for _ in range(2000):
            si = rng.integers(len(seeds), size=len(seeds))
            qi = rng.integers(delta.shape[1], size=delta.shape[1])
            draws.append(float(delta[np.ix_(si, qi)].mean() * 100))
        intervals[f'{method}_minus_{reference}'] = dict(mean_pp=float(delta.mean()*100),
            seed_differences_pp=(delta.mean(1)*100).tolist(),
            descriptive_95_interval_pp=np.quantile(draws, [.025, .975]).tolist())
    health_path = out / 'ppo-health.json'
    if health_path.exists():
        health = json.loads(health_path.read_text())
        overhead = np.mean([health[str(seed)]['extra_forward_wall_seconds_estimate'] for seed in seeds]) * 2 / 3600
        stats['ppo']['diagnostic_gpu_hours_estimate_mean'] = float(overhead)
        stats['ppo']['gpu_hours_excluding_diagnostic_forward_estimate_mean'] = stats['ppo']['gpu_hours_mean'] - float(overhead)
    result = dict(methods=stats, paired=intervals,
        scope='Paired seed/question bootstrap; only three seeds by default. Inspected test split; exploratory intervals, no superiority gate.')
    write_json(out / 'aggregate.json', result)
    with (out / 'report.txt').open('a') as handle:
        handle.write('\nAcross training seeds (accuracy mean +/- sample SD; GPU hours mean):\n')
        for method, row in stats.items():
            handle.write(f"{method}: {100*row['accuracy_mean']:.2f}% +/- {100*row['accuracy_seed_sd']:.2f} pp; {row['gpu_hours_mean']:.3f} GPUh\n")
        if 'diagnostic_gpu_hours_estimate_mean' in stats['ppo']:
            handle.write(f"PPO diagnostic forward overhead estimate: {stats['ppo']['diagnostic_gpu_hours_estimate_mean']:.3f} GPUh/run; "
                         f"training excluding that estimate: {stats['ppo']['gpu_hours_excluding_diagnostic_forward_estimate_mean']:.3f} GPUh/run.\n")
        handle.write(result['scope'] + '\n')
    return result


def run(args):
    if args.steps < 1 or args.calibration_steps < 1 or len(set(args.seeds)) != len(args.seeds) or len(args.seeds) < 3:
        raise ValueError('Use positive budgets and at least three unique final seeds')
    if 17 in args.seeds:
        raise ValueError('Seed 17 is reserved for calibration')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    state = dict(state='preparing')
    write_json(out / 'status.json', state)
    try:
        data = comparison.prepare(args.source, out)
        manifest = dict(protocol='math-online-calibrated-v1', data=data, seeds=args.seeds,
            methods=list(comparison.METHODS), steps=args.steps, answers_per_update=64,
            linear_critic=dict(feature_mode='actor', buffer_capacity=0),
            calibration=dict(seed=17, steps=args.calibration_steps, candidates=PPO_PROFILES,
                selection='final validation accuracy among healthy, non-regressing candidates; declared-order tie-break'),
            primary_comparison='lstd versus calibrated ppo', provenance=snapshot(comparison.ROOT))
        write_json(out / 'manifest.json', manifest)
        candidates = []
        for profile in PPO_PROFILES:
            name = f'calibration-{profile}'
            state.update(state='calibrating', current=name)
            write_json(out / 'status.json', state)
            result = launch(out, 'ppo', 17, args.calibration_steps, profile, True, name)
            result.update(profile=profile, health=audit_health(out / name / 'ppo-critic-audit', args.calibration_steps))
            candidates.append(result)
            write_json(out / 'calibration.json', dict(candidates=candidates))
        profile = select_profile(candidates)
        tuning_gpu_hours = sum(r['training_gpu_hours'] for r in candidates)
        write_json(out / 'selection.json', dict(profile=profile, candidates=candidates,
            tuning_training_gpu_hours=tuning_gpu_hours,
            selection_data='validation only; no final test scores read'))
        manifest['selected_ppo_profile'] = profile
        write_json(out / 'manifest.json', manifest)
        for seed in args.seeds:
            # Rotate order to reduce systematic method/runtime order confounding.
            offset = args.seeds.index(seed) % len(comparison.METHODS)
            order = comparison.METHODS[offset:] + comparison.METHODS[:offset]
            for method in order:
                state.update(state='comparing', current=f'{method}-seed{seed}')
                write_json(out / 'status.json', state)
                launch(out, method, seed, args.steps, profile if method == 'ppo' else 'legacy')
        summary = comparison.report(out)
        if not summary['complete']:
            raise RuntimeError('Incomplete comparison')
        if not all(c['identical_base_scores'] for c in summary['comparisons'].values()):
            raise RuntimeError('Different starting-model test scores within a seed; inspect before comparing methods')
        health = {str(seed): audit_health(out / f'ppo-seed{seed}/ppo-critic-audit', args.steps) for seed in args.seeds}
        write_json(out / 'ppo-health.json', health)
        aggregate(out, args.seeds)
        with (out / 'report.txt').open('a') as handle:
            handle.write(f'PPO validation selection: {profile}; additional tuning training cost: {tuning_gpu_hours:.3f} GPUh.\n')
            handle.write(f'PPO critic health checks: {all(h["pass_checks"] for h in health.values())}. See ppo-health.json.\n')
        state.update(state='complete' if all(h['pass_checks'] for h in health.values()) else 'needs_review')
        write_json(out / 'status.json', state)
    except BaseException as exc:
        state.update(state='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(out / 'status.json', state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--calibration-steps', type=int, default=30)
    parser.add_argument('--steps', type=int, default=60)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    run(parser.parse_args())


if __name__ == '__main__':
    main()
