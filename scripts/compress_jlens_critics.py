"""CPU-only PCA study reusing completed large J-lens sufficient statistics.

All fits and validation selections finish before test loading. Source is read-only.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
from math_rl.critic_compression import pca_from_statistics, affine_projection, project_statistics
from math_rl.large_linear import solve
from math_rl.large_probe import paired_errors
from math_rl.critic_probe import metrics
from math_rl.lstd import design
from math_rl.provenance import sha256
from large_jlens_study import dataset, read, write
import frozen_critics as frozen

ARMS = ('direct_jlens_concat', 'hidden_concat', 'rms_hidden_concat')
BASELINES = ('hidden_two_thirds', 'final')
HEADS = ('ridge', 'lstd-0.99')


def merge(source, arm, shards):
    merged = None
    normalization = None
    for shard in range(shards):
        path = source / 'linear-statistics' / f'shard-{shard}.pt'
        record = torch.load(path, weights_only=True, map_location='cpu')
        current = record['stats'][arm]
        if merged is None:
            merged, normalization = current, record['normalization'][arm]
        else:
            if any(not torch.equal(a, b) for a, b in zip(normalization, record['normalization'][arm])):
                raise ValueError('Source shards use different feature coordinates')
            for key in merged:
                merged[key] += current[key]
    return merged, normalization


def predict(data, mean, scale, weights):
    # Fold the fixed projection into weights; avoids huge projected evaluation caches.
    return torch.cat([design(x, mean, scale) @ weights for x in data['x'].split(2048)]).numpy()


def check_labels(reference, data):
    labels = {k: data[k] for k in ('y', 'question', 'response', 'position')}
    if reference is not None and any(not torch.equal(reference[k], labels[k]) for k in labels):
        raise ValueError('Unmatched evaluation prefixes')
    return labels


def run(args):
    source, out = args.source.resolve(), args.out.resolve()
    if source == out or source in out.parents:
        raise ValueError('Use a separate output directory outside the source study')
    m = read(source / 'manifest.json')
    if m['config']['linear_gamma'] != 1 or m['config']['linear_lambda'] != .99:
        raise ValueError('Requires the completed gamma=1, lambda=.99 study')
    ranks = sorted(set(args.dimensions))
    if not ranks or min(ranks) < 1:
        raise ValueError('Dimensions must be positive')
    shards = m['config']['shards']
    files = [source / 'manifest.json']
    for shard in range(shards):
        path = source / 'linear-statistics' / f'shard-{shard}.pt'
        if sha256(path) != read(path.with_suffix('.json'))['sha256']:
            raise ValueError('Source statistics hash mismatch')
        files.append(path)
    files += [source / 'packed' / arm / 'complete.json' for arm in ARMS + BASELINES]
    protocol = dict(source=str(source), dimensions=ranks, epsilon=m['config']['linear_epsilon'],
        inputs={str(p.relative_to(source)): sha256(p) for p in files},
        code={str(p.relative_to(ROOT)): sha256(p) for p in
              (Path(__file__).resolve(), ROOT/'src/math_rl/critic_compression.py',
               ROOT/'src/math_rl/large_linear.py', ROOT/'scripts/large_jlens_study.py',
               ROOT/'src/math_rl/lstd.py', ROOT/'src/math_rl/large_probe.py',
               ROOT/'src/math_rl/critic_probe.py')},
        pca='All-token training covariance; centered, no whitening; original isotropic scaling retained',
        selection='Lowest validation Brier per arm/head; includes uncompressed candidate; no test selection')
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'protocol.json').exists() and read(out / 'protocol.json') != protocol:
        raise ValueError('Protocol/source changed; use a new output directory')
    write(out / 'protocol.json', protocol)
    selection = {}
    reference = None
    for arm in ARMS + BASELINES:
        path = out / f'{arm}.pt'
        if path.exists():
            if sha256(path) != read(path.with_suffix('.json'))['sha256']:
                raise ValueError('Saved fit changed')
            record = torch.load(path, weights_only=True)
        else:
            start = time.perf_counter()
            stats, (mean, scale) = merge(source, arm, shards)
            dimension = len(mean)
            if arm in ARMS and max(ranks) >= dimension:
                raise ValueError('Compression dimensions must be smaller than source width')
            val = dataset(source, arm, 'val')
            reference = check_labels(reference, val)
            candidates = {}
            center = eigenvalues = components = None
            if arm in ARMS:
                center, eigenvalues, vectors = pca_from_statistics(stats)
                components = vectors[:, :max(ranks)].clone()
                del vectors
            for rank in ([dimension] + ranks if arm in ARMS else [dimension]):
                print(f'Fitting {arm}, dimension={rank}', flush=True)
                transform = None if rank == dimension else affine_projection(center, components[:, :rank])
                reduced = stats if transform is None else project_statistics(stats, transform)
                for head in HEADS:
                    weights, checks = solve(reduced, head, protocol['epsilon'])
                    effective = weights if transform is None else transform @ weights
                    values = predict(val, mean, scale, effective)
                    candidates[f'{rank}/{head}'] = dict(weights=weights, effective_weights=effective,
                        checks=checks, validation=metrics(val['y'].numpy(), values),
                        variance_retained=1. if transform is None else float(eigenvalues[:rank].sum()/eigenvalues.sum()))
                if transform is not None:
                    del reduced
            record = dict(candidates=candidates, mean=mean, scale=scale, center=center,
                components=components, eigenvalues=eigenvalues, original_dimension=dimension,
                seconds=time.perf_counter()-start)
            frozen.save_tensor(path, record)
            write(path.with_suffix('.json'), dict(sha256=sha256(path)))
            del stats, val
        for head in HEADS:
            eligible = {k: v for k, v in record['candidates'].items() if k.endswith('/'+head)}
            best = min(eligible, key=lambda k: eligible[k]['validation']['brier'])
            selection[f'{arm}/{head}'] = dict(candidate=best,
                validation_brier=eligible[best]['validation']['brier'])
        del record
    # Lock every validation selection before opening test features/labels.
    write(out / 'selection.json', selection)
    predictions, results = {}, {}
    reference = None
    for arm in ARMS + BASELINES:
        record = torch.load(out / f'{arm}.pt', weights_only=True)
        data = dataset(source, arm, 'test')
        reference = check_labels(reference, data)
        for key, candidate in record['candidates'].items():
            name = f'{arm}/{key}'
            values = predict(data, record['mean'], record['scale'], candidate['effective_weights'])
            predictions[name] = values[None]
            results[name] = dict(test=metrics(data['y'].numpy(), values),
                validation=candidate['validation'], variance_retained=candidate['variance_retained'])
        del record, data
    y, questions = reference['y'].numpy(), reference['question'].numpy()
    comparisons = {}
    for rank in ranks:
        for head in HEADS:
            key = f'direct_jlens_concat/{rank}/{head}'
            for other in ('hidden_concat', 'rms_hidden_concat'):
                control = f'{other}/{rank}/{head}'
                comparisons[f'{key}_minus_{control}'] = paired_errors(y, predictions[key], predictions[control], questions)
            for other in BASELINES:
                control = next(k for k in predictions if k.startswith(other+'/') and k.endswith('/'+head))
                comparisons[f'{key}_minus_{control}'] = paired_errors(y, predictions[key], predictions[control], questions)
    for arm in ARMS:
        for head in HEADS:
            full = next(k for k in predictions if k.startswith(arm+'/') and k.endswith('/'+head)
                        and int(k.split('/')[1]) not in ranks)
            for rank in ranks:
                key = f'{arm}/{rank}/{head}'
                comparisons[f'{key}_minus_full'] = paired_errors(y, predictions[key], predictions[full], questions)
    scope = ('Exploratory compression study on an already inspected test set. PCA uses all-token training Gram only; '
             'raw-sum Ridge/LSTD projected exactly before epsilon*I. Dimension selected on validation only. '
             'All predefined dimensions reported; paired question intervals unadjusted, no superiority gate. '
             'Retained verifier labels; no new generation, PPO update, or end-to-end cost claim.')
    write(out / 'summary.json', dict(scope=scope, results=results, comparisons=comparisons, selection=selection))
    np.savez_compressed(out/'test-predictions.npz', **{k: v[0] for k, v in predictions.items()},
                        y=y, question=questions)
    lines = [scope, '', 'representation/dimension/head                 accuracy    Brier     variance retained']
    for key, value in results.items():
        t = value['test']
        lines.append(f'{key:48s} {t["accuracy"]:.4f}    {t["brier"]:.5f}    {value["variance_retained"]:.4f}')
    lines += ['', 'Validation selections:', json.dumps(selection, indent=2), '', 'Paired Brier differences (negative favors first):']
    lines += [f'{k}: {v}' for k, v in comparisons.items()]
    (out/'report.txt').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--dimensions', nargs='+', type=int, default=[768, 1536, 3072])
    args = parser.parse_args()
    if args.source.resolve() == args.out.resolve() or args.source.resolve() in args.out.resolve().parents:
        parser.error('Output must be outside the source study')
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '8')))
    start = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        write(args.out/'status.json', dict(state='running'))
        run(args)
    except Exception as exc:
        write(args.out/'status.json', dict(state='failed', error=str(exc), seconds=time.perf_counter()-start))
        raise
    write(args.out/'status.json', dict(state='complete', seconds=time.perf_counter()-start))


if __name__ == '__main__':
    main()
