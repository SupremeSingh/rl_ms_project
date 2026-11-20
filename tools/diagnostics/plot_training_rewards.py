"""Plot saved training rewards; never modify the source study or infer missing updates."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import mean, stdev

NUMBER = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
STEP = re.compile(r'\bstep:(\d+)\s+-')
SCORE = re.compile(r'\bcritic/score/mean:\s*(' + NUMBER + r')')
ANSI = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')


def logged_rewards(path):
    records = {}
    for line in path.read_text(errors='replace').splitlines():
        line = ANSI.sub('', line)
        step, score = STEP.search(line), SCORE.search(line)
        if not step or not score:
            continue
        update, value = int(step[1]), float(score[1])
        if update < 1:
            continue
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'Invalid binary-reward mean: {path}, update {update}')
        if update in records and records[update] != value:
            raise ValueError(f'Conflicting repeated update: {path}, update {update}')
        records[update] = value
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--window', type=int, default=10)
    a = p.parse_args()
    source, out = a.study.resolve(), a.out.resolve()
    if out == source or source in out.parents:
        raise ValueError('Use a separate output directory outside the source study')
    manifest_path = source / 'manifest.json'
    m = json.loads(manifest_path.read_text())
    verification = json.loads((source / 'verification.json').read_text())
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if not verification['passed'] or verification['manifest_sha256'] != digest:
        raise ValueError('Require matching completed-study verification')
    arms, seeds, steps = m['curve_arms'], m['seeds'], m['steps']
    if not 1 <= a.window <= steps or len(seeds) < 2:
        raise ValueError('Invalid smoothing window or seed count')
    expected = set(range(1, steps + 1))
    rewards, coverage = {}, {}
    for arm in arms:
        rewards[arm] = {}
        for seed in seeds:
            case = source / 'runs' / f'{arm}-seed{seed}'
            if json.loads((case / 'result.json').read_text())['updates'] != steps:
                raise ValueError(f'Incomplete source run: {case}')
            records = logged_rewards(case.with_suffix('.log'))
            if set(records) - expected:
                raise ValueError(f'Unexpected updates: {case}')
            rewards[arm][str(seed)] = records
            missing = sorted(expected - records.keys())
            coverage[f'{arm}-seed{seed}'] = dict(recorded=len(records), missing=missing)
            print(f'{arm:<23} seed {seed}: {len(records)}/{steps}; missing {missing}')
    # A plotted point requires the entire trailing window in every run.
    endpoints = [i for i in range(a.window, steps + 1)
                 if all(all(j in r for j in range(i-a.window+1, i+1))
                        for runs in rewards.values() for r in runs.values())]
    if not endpoints:
        raise ValueError('No fully matched windows; inspect logs before plotting')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(9, 4.7))
    labels = {'slice-token': 'SLICE/token', 'slice-chunk': 'SLICE/chunk',
              'ppo': 'PPO', 'grpo': 'GRPO', 'rms-layer-token': 'Normalized layer/token'}
    colors = ['#2765A7', '#A064AA', '#CC782A', '#248A67', '#858B93']
    plotted = {}
    for arm, color in zip(arms, colors):
        ys = [[100 * mean(rewards[arm][str(seed)][j]
                          for j in range(i-a.window+1, i+1)) for i in endpoints]
              for seed in seeds]
        center = [mean(v) for v in zip(*ys)]
        sd = [stdev(v) for v in zip(*ys)]
        ax.plot(endpoints, center, label=labels.get(arm, arm), color=color, lw=2)
        ax.fill_between(endpoints, [v-s for v, s in zip(center, sd)],
                        [v+s for v, s in zip(center, sd)], color=color, alpha=.13)
        plotted[arm] = dict(mean_percent=center, seed_sd_pp=sd)
    ax.set(xlabel='Training update', ylabel='Training rollout success (%)',
           title=f'{len(seeds)} seeds · {a.window}-update trailing mean ± seed SD')
    ax.set_xlim(0, steps)
    ax.grid(alpha=.2)
    ax.legend(fontsize=10, frameon=True)
    fig.text(.5, .025, 'Stochastic on-policy training answers; not greedy validation/test accuracy.',
             ha='center', fontsize=9, color='#555555')
    fig.subplots_adjust(left=.10, right=.98, top=.88, bottom=.20)
    out.mkdir(parents=True, exist_ok=False)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(out / f'training-reward.{ext}', dpi=220, bbox_inches='tight')
    plt.close(fig)
    (out / 'training-reward.json').write_text(json.dumps(dict(
        source=str(source), manifest_sha256=digest, metric='critic/score/mean',
        scope='Rounded console metrics; trailing window per seed then mean/sample SD. No missing updates filled.',
        seeds=seeds, window=a.window, endpoints=endpoints, coverage=coverage,
        raw_rewards=rewards, plotted=plotted), indent=2) + '\n')
    print(f'\nSaved: {out}/training-reward.png')
    print('Missing console records are reported and never imputed.')


if __name__ == '__main__':
    main()
