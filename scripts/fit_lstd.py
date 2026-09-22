"""Fit a single affine value head with LSTD(0) and matched ridge on saved trajectories."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
from math_rl.critic_probe import assessment
from math_rl.lstd import accumulate, empty_statistics, predict, solve
from math_rl.provenance import sha256, write_json


def save(path, value):
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def read_shard(source, index, question):
    feature_path = source / 'features' / f'{index:05d}.pt'
    trajectory_path = source / 'trajectories' / f'{index:05d}.json'
    shard = torch.load(feature_path, map_location='cpu', weights_only=True)
    raw = json.loads(trajectory_path.read_text())
    if (shard['question_id'] != question['id'] or raw['question_id'] != question['id']
            or shard['trajectories_sha256'] != sha256(trajectory_path)):
        raise ValueError('Feature/trajectory provenance mismatch')
    indices = shard['response_indices'].tolist()
    offsets = shard['offsets'].tolist()
    if (len(indices) != len(shard['rewards']) or len(offsets) != len(indices) + 1
            or offsets[0] != 0 or offsets[-1] != len(shard['features'])
            or len(set(indices)) != len(indices)):
        raise ValueError('Invalid feature offsets or response indices')
    expected = [j for j, row in enumerate(raw['responses'])
                if row['verifier_status'] in ('correct', 'incorrect') and row['token_ids']]
    if indices != expected:
        raise ValueError('Cached responses do not match the declared exclusion policy')
    for j, response_index in enumerate(indices):
        row = raw['responses'][response_index]
        if (row['score'] != float(shard['rewards'][j]) or row['score'] not in (0, 1)
                or row['finish_reason'] not in ('stop', 'length')
                or offsets[j + 1] - offsets[j] != len(row['token_ids']) + 1):
            raise ValueError('Invalid reward, terminal boundary or feature alignment')
    return shard, raw, dict(features=sha256(feature_path), trajectories=sha256(trajectory_path))


def validate_probe(data, questions, split, dimension):
    allowed = {i for i, q in enumerate(questions) if q['split'] == split}
    if (data['x'].ndim != 2 or data['x'].shape[1] != dimension or not len(data['y'])
            or any(len(data[k]) != len(data['y']) for k in ('x', 'question', 'position'))
            or not set(data['question'].tolist()).issubset(allowed)
            or not torch.isfinite(data['x']).all()
            or not ((data['y'] == 0) | (data['y'] == 1)).all()):
        raise ValueError(f'Invalid {split} probe cache or question split')


def run(source, out, alphas):
    source, out = source.resolve(), out.resolve()
    if source == out or source in out.parents:
        raise ValueError('Use a separate output directory; source artifacts are read-only')
    if not alphas or any(not np.isfinite(a) or a <= 0 for a in alphas):
        raise ValueError('Use positive finite regularization candidates')
    names = ('manifest.json', 'questions.json', 'summary.json', 'normalization.pt', 'val.pt', 'test.pt')
    hashes = {name: sha256(source / name) for name in names}
    source_manifest = json.loads((source / 'manifest.json').read_text())
    if source_manifest['questions_sha256'] != hashes['questions.json']:
        raise ValueError('Question manifest changed')
    if source_manifest['config']['protocol'] != 'frozen-critics-v2':
        raise ValueError('Expected frozen-critics-v2 feature cache')
    questions = json.loads((source / 'questions.json').read_text())['questions']
    ids = [q['id'] for q in questions]
    if len(ids) != len(set(ids)) or set(q['split'] for q in questions) != {'train', 'val', 'test'}:
        raise ValueError('Invalid question splits')
    config = dict(protocol='frozen-lstd-v1', source=str(source), source_hashes=hashes,
        alphas=alphas, gamma=1., trace_lambda=0., weighting='uniform over retained training transitions',
        terminal='EOS and length cap both terminal; final action gets stored outcome; no bootstrap',
        exclusions='Preserve source: only correct/incorrect nonempty answers; conditional evaluation',
        features='same normalized Qwen hidden vector plus intercept; raw linear output, no sigmoid',
        versions=dict(torch=str(torch.__version__), numpy=np.__version__),
        code={str(p.relative_to(ROOT)): sha256(p) for p in
              (Path(__file__).resolve(), ROOT / 'src/math_rl/lstd.py', ROOT / 'src/math_rl/critic_probe.py')})
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / 'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != config:
            raise ValueError('Resume requires unchanged inputs, code, settings and runtime')
    else:
        if any(out.iterdir()):
            raise ValueError('Output must be empty or a compatible LSTD run')
        write_json(manifest_path, config)
    norm = torch.load(source / 'normalization.pt', weights_only=True, map_location='cpu')
    mean, scale = norm['mean'].double(), norm['scale'].double()
    if (mean.ndim != 1 or mean.shape != scale.shape or not torch.isfinite(mean).all()
            or not torch.isfinite(scale).all() or (scale <= 0).any()):
        raise ValueError('Invalid training normalization')
    checkpoint = out / 'statistics.pt'
    if checkpoint.exists():
        saved = torch.load(checkpoint, weights_only=True, map_location='cpu')
    else:
        saved = dict(stats=empty_statistics(len(mean)), processed={}, seconds=0., statuses={}, capped=0)
    stats = saved['stats']
    started = time.perf_counter()
    for i, q in enumerate(questions):
        if q['split'] != 'train':
            continue
        key = str(i)
        if key in saved['processed']:
            actual = dict(features=sha256(source / 'features' / f'{i:05d}.pt'),
                          trajectories=sha256(source / 'trajectories' / f'{i:05d}.json'))
            if actual != saved['processed'][key]:
                raise ValueError('Previously accumulated source shard changed')
            continue
        shard, raw, fingerprint = read_shard(source, i, q)
        if shard.get('exclusion_policy') != source_manifest['config']['exclusion_policy']:
            raise ValueError('Exclusion policy mismatch')
        counts = Counter(saved['statuses'])
        counts.update(r['verifier_status'] for r in raw['responses'])
        saved['statuses'] = dict(counts)
        for j, reward in enumerate(shard['rewards'].tolist()):
            lo, hi = shard['offsets'][j:j + 2].tolist()
            accumulate(stats, shard['features'][lo:hi], reward, mean, scale)
            saved['capped'] += raw['responses'][int(shard['response_indices'][j])]['finish_reason'] == 'length'
        saved['processed'][key] = fingerprint
        if len(saved['processed']) % 25 == 0:
            saved['seconds'] += time.perf_counter() - started
            save(checkpoint, saved)
            started = time.perf_counter()
            print(f"Accumulated {len(saved['processed'])} training questions, {stats['transitions']} transitions", flush=True)
    saved['seconds'] += time.perf_counter() - started
    save(checkpoint, saved)
    val = torch.load(source / 'val.pt', weights_only=True, map_location='cpu')
    validate_probe(val, questions, 'val', len(mean))
    # Select both methods before opening test tensors. Same transitions and alpha grid.
    selected, trials = {}, {}
    for method in ('lstd', 'ridge'):
        trials[method] = []
        for alpha in alphas:
            started = time.perf_counter()
            try:
                weights, diagnostic = solve(stats, method, alpha)
                prediction = predict(weights, val, mean, scale)
                error = float(np.mean((prediction - val['y'].numpy()) ** 2))
                if not np.isfinite(error) or diagnostic['relative_residual'] > 1e-10:
                    raise ValueError('Nonfinite validation loss or inaccurate solve')
            except (torch.linalg.LinAlgError, ValueError) as exc:
                trials[method].append(dict(alpha=alpha, error=str(exc), seconds=time.perf_counter() - started))
                continue
            trial = dict(alpha=alpha, validation_brier=error, **diagnostic, seconds=time.perf_counter() - started)
            trials[method].append(trial)
            if method not in selected or error < selected[method]['validation_brier']:
                selected[method] = dict(**trial, weights=weights)
        if method not in selected:
            raise RuntimeError(f'No valid {method} solve: {trials[method]}')
        winner = selected[method]
        matrix = stats['a' if method == 'lstd' else 'gram'] / stats['transitions']
        penalty = torch.eye(len(mean) + 1, dtype=torch.float64)
        penalty[-1, -1] = 0
        winner['condition_number'] = float(torch.linalg.cond(matrix + winner['alpha'] * penalty))
        weights = winner['weights']
        save(out / f'{method}.pt', dict(**winner, mean=mean, scale=scale,
             state=dict(weight=weights[:-1].float()[None], bias=weights[-1:].float())))
    test = torch.load(source / 'test.pt', weights_only=True, map_location='cpu')
    validate_probe(test, questions, 'test', len(mean))
    results, predictions = {}, {}
    for method, winner in selected.items():
        predictions[method] = predict(winner['weights'], test, mean, scale)
        np.save(out / f'{method}-predictions.npy', predictions[method])
        results[method] = dict(**{k: v for k, v in winner.items() if k != 'weights'},
                              trials=trials[method], test=assessment(test, predictions[method], norm['baseline']))
    difference = (predictions['lstd'] - test['y'].numpy()) ** 2 - (predictions['ridge'] - test['y'].numpy()) ** 2
    _, groups = np.unique(test['question'].numpy(), return_inverse=True)
    per_question = np.bincount(groups, weights=difference) / np.bincount(groups)
    rng = np.random.default_rng(42)
    interval = np.quantile([rng.choice(per_question, len(per_question), replace=True).mean()
                           for _ in range(1000)], [.025, .975]).tolist()
    report = dict(scope='Frozen-policy LSTD(0), single affine value head. No actor or PPO update.',
        training_transitions=stats['transitions'], retained_training_answers=stats['trajectories'],
        training_verifier_status_counts=saved['statuses'], retained_capped_answers=saved['capped'],
        shared_accumulation_seconds=saved['seconds'], methods=results,
        paired_question_brier_lstd_minus_ridge=float(per_question.mean()), paired_95pct_interval=interval,
        evaluation='Existing eight-prefix-per-answer held-out protocol; training uses all transitions. '
                   'Conditional on retained answers. This test set was already inspected in the earlier study.',
        limitations='Exclusion selection bias; cap treated as terminal; regularization shifts the TD fixed point; '
                    'a hidden vector need not be Markov. No online learning claim.')
    write_json(out / 'summary.json', report)
    lines = ['Frozen-policy LSTD(0) versus matched ridge; single linear value head',
             f"Training: {stats['trajectories']} retained answers, {stats['transitions']} transitions",
             'method   alpha       test Brier   accuracy   condition number']
    for method, result in results.items():
        lines.append(f"{method:8} {result['alpha']:<11g} {result['test']['brier']:.5f}      "
                     f"{result['test']['accuracy']:.3f}      {result['condition_number']:.3g}")
    lines += [f'Paired question Brier difference LSTD minus ridge: {per_question.mean():.5f}; 95% interval {interval}',
              'Negative favors LSTD. See summary.json for calibration, solver checks, costs and limitations.',
              'Source data unchanged; excludes unparseable answers. No PPO learning claim.']
    (out / 'report.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--alphas', type=float, nargs='+', default=[1e-6, 1e-5, 1e-4, .001, .01, .1, 1.])
    args = parser.parse_args()
    run(args.source, args.out, args.alphas)


if __name__ == '__main__':
    main()
