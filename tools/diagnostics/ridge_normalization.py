"""Offline three-arm Ridge ablation on existing matched Numina prefix caches.

No generation, lens, encoder, compressor training, calibration or PPO updates.
LayerNorm is per-prefix, after ReLU, without learnable affine parameters.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from slice_critic.features import rms_coordinates
from slice_critic.inference import load_filter
from slice_critic.storage import sha256, write_json

ARMS = ('rms_two_thirds', 'slice_layernorm', 'slice_baseline')
POLICIES = ('legacy', 'relative')


def transform(arm, x, fitted, rms_eps, layernorm_eps):
    if arm == 'rms_two_thirds':
        return rms_coordinates(x.float(), rms_eps)
    model, mean, scale = fitted
    z = model.features((x.float() - mean) / scale)
    if arm == 'slice_layernorm':
        return F.layer_norm(z, (z.shape[1],), eps=layernorm_eps)
    if arm != 'slice_baseline':
        raise ValueError('Unknown arm')
    return z


def ridge_fit(a, b, policy, strength, *, epsilon=None):
    # Raw sums retained. Relative policy: fixed fractional loading of mean diagonal.
    if epsilon is None:
        if policy not in POLICIES:
            raise ValueError('Unknown Ridge regularization policy')
        epsilon = .01 if policy == 'legacy' else max(.01, strength * float(a.trace()) / len(b))
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('Ridge epsilon must be finite and positive')
    system = a + epsilon * torch.eye(len(b), dtype=a.dtype)
    w = torch.linalg.solve(system, b)
    residual = torch.linalg.vector_norm(system @ w - b) / (
        torch.linalg.matrix_norm(system) * torch.linalg.vector_norm(w) + torch.linalg.vector_norm(b)
    ).clamp_min(1e-30)
    if not torch.isfinite(w).all() or residual > 1e-8:
        raise ValueError('Ridge solve failed')
    eigenvalues = torch.linalg.eigvalsh(system)
    return w, dict(epsilon=epsilon, relative_residual=float(residual),
                   condition_2=float(eigenvalues[-1] / eigenvalues[0]),
                   weight_norm=float(w.norm()))


def open_split(source, split):
    datasets = {}
    for arm in ('rms_hidden_concat', 'hidden_two_thirds'):
        folder = source / 'packed' / arm
        complete = json.loads((folder / 'complete.json').read_text())
        for suffix in ('x.npy', 'labels.pt'):
            path = folder / f'{split}-{suffix}'
            if sha256(path) != complete['files'][path.name]:
                raise ValueError(f'Packed feature checksum mismatch: {path}')
        metadata = torch.load(folder / f'{split}-labels.pt', map_location='cpu', weights_only=True)
        x = np.load(folder / f'{split}-x.npy', mmap_mode='c')
        if x.ndim != 2 or len(x) != len(metadata['y']) or not len(x):
            raise ValueError('Empty or mismatched feature rows')
        datasets[arm] = (x, metadata)
    left, right = (datasets[a][1] for a in datasets)
    for field in ('y', 'question', 'response', 'position'):
        if not torch.equal(left[field], right[field]):
            raise ValueError(f'Unpaired cached states: {split}/{field}')
    if set(left['y'].tolist()) != {0., 1.}:
        raise ValueError('Each split requires both outcome classes')
    if datasets['rms_hidden_concat'][0].shape[1] != 7680 or datasets['hidden_two_thirds'][0].shape[1] != 1536:
        raise ValueError('Unexpected feature dimensions')
    return datasets, left


def batch_features(datasets, arm, start, stop, fitted, eps, ln_eps):
    key = 'hidden_two_thirds' if arm == 'rms_two_thirds' else 'rms_hidden_concat'
    x = torch.from_numpy(datasets[key][0][start:stop])
    if not torch.isfinite(x).all():
        raise ValueError('Nonfinite cached features')
    z = transform(arm, x, fitted, eps, ln_eps).double()
    return torch.cat([z, torch.ones(len(z), 1, dtype=z.dtype)], 1)


def evaluate(y, predictions, questions):
    if not np.isfinite(predictions).all():
        raise ValueError('Nonfinite Ridge predictions')
    squared = (predictions - y) ** 2
    ids, inverse = np.unique(questions, return_inverse=True)
    grouped = np.bincount(inverse, weights=squared) / np.bincount(inverse)
    return dict(prefixes=len(y), questions=len(ids), raw_brier=float(squared.mean()),
                question_mean_raw_brier=float(grouped.mean()),
                clipped_brier=float(((np.clip(predictions, 0, 1) - y) ** 2).mean()),
                accuracy=float(((predictions >= .5) == y).mean()),
                outside_unit_interval=float(((predictions < 0) | (predictions > 1)).mean())), grouped


@torch.inference_mode()
def run(args):
    source, study, out = args.features.resolve(), args.study.resolve(), args.out.resolve()
    if any(out == p or p in out.parents for p in (source, study)):
        raise ValueError('Output must be separate from inputs')
    if out.exists():
        raise ValueError('Use a new output directory')
    extra_epsilons = getattr(args, 'ridge_epsilons', [])
    if any(not np.isfinite(e) or e <= 0 for e in extra_epsilons):
        raise ValueError('Ridge epsilons must be finite and positive')
    policies = dict.fromkeys(POLICIES)
    policies.update({f'fixed_{float(e):.12g}': float(e) for e in extra_epsilons})
    source_manifest = json.loads((source / 'manifest.json').read_text())
    study_manifest = json.loads((study / 'manifest.json').read_text())
    if source_manifest['model_hashes'] != study_manifest['model_hashes']:
        raise ValueError('Feature source and frozen filter model identities differ')
    if source_manifest['layers'] != study_manifest['layers'] or source_manifest['rms_eps'] != study_manifest['rms_eps']:
        raise ValueError('Source layer order or normalization differs')
    filter_info = study_manifest['filters']['sparse']
    path = study / 'filters' / Path(filter_info['path']).name
    fitted = load_filter(path, filter_info['sha256'], layers=study_manifest['layers'], eps=study_manifest['rms_eps'])
    splits, metadata, question_sets = {}, {}, []
    for split in ('train', 'val', 'test'):
        splits[split], metadata[split] = open_split(source, split)
        question_sets.append(set(metadata[split]['question'].tolist()))
    if any(question_sets[i] & question_sets[j] for i in range(3) for j in range(i)):
        raise ValueError('Question splits overlap')
    out.mkdir(parents=True)
    started = time.perf_counter()
    protocol = dict(arms=list(ARMS), strength=args.strength, legacy_epsilon=.01,
        regularization_policies=policies,
        relative_rule='max(0.01, strength * trace(A_train) / design_width)',
        intercept_penalized=True, layernorm_eps=args.layernorm_eps, affine=False,
        normalized_two_thirds='Per-prefix RMS, no learned gain, no dataset centering',
        slice_layernorm='Frozen existing SLICE ReLU output, then per-prefix LayerNorm',
        source=str(source), study=str(study), filter_sha256=filter_info['sha256'],
        script_sha256=sha256(Path(__file__)), source_manifest_sha256=sha256(source/'manifest.json'),
        study_manifest_sha256=sha256(study/'manifest.json'),
        cache_manifests={a:sha256(source/'packed'/a/'complete.json')
                         for a in ('rms_hidden_concat', 'hidden_two_thirds')},
        scope='Frozen representations; same saved sampled prefixes/labels in every arm. '
              'Inspected Numina splits and previously validation-selected compressor. '
              'Not fresh data, all-token fitting, or an RL comparison. '
              'Primary regularization policy: relative. Legacy epsilon is a declared control. '
              'No hyperparameter selection on these results; no calibration.')
    write_json(out/'protocol.json', protocol)
    weights, fits = {}, {}
    for arm in ARMS:
        k = 1537 if arm == 'rms_two_thirds' else 769
        a, b = torch.zeros(k, k, dtype=torch.float64), torch.zeros(k, dtype=torch.float64)
        for start in range(0, len(metadata['train']['y']), args.batch_size):
            stop = start + args.batch_size
            x = batch_features(splits['train'], arm, start, stop, fitted, study_manifest['rms_eps'], args.layernorm_eps)
            y = metadata['train']['y'][start:stop].double()
            a += x.T @ x
            b += x.T @ y
        for policy, epsilon in policies.items():
            key = f'{arm}/{policy}'
            weights[key], fits[key] = ridge_fit(a, b, policy, args.strength, epsilon=epsilon)
        torch.save(dict(a=a, b=b, training_prefixes=len(metadata['train']['y'])),
                   out/f'{arm}-statistics.pt')
        print(f'Fitted {arm}', flush=True)
    torch.save(dict(weights=weights, fits=fits, protocol=protocol), out/'heads.pt')
    results, test_errors = {}, {}
    # Validation is evaluated before test. It does not select a policy or arm.
    for split in ('val', 'test'):
        results[split] = {}
        for arm in ARMS:
            predictions = {p:[] for p in policies}
            for start in range(0, len(metadata[split]['y']), args.batch_size):
                x = batch_features(splits[split], arm, start, start+args.batch_size, fitted,
                                   study_manifest['rms_eps'], args.layernorm_eps)
                for policy in policies:
                    predictions[policy].append((x @ weights[f'{arm}/{policy}']).numpy())
            for policy in policies:
                key = f'{arm}/{policy}'
                pred = np.concatenate(predictions[policy])
                results[split][key], grouped = evaluate(metadata[split]['y'].numpy(), pred,
                                                       metadata[split]['question'].numpy())
                np.save(out/f'{split}-{arm}-{policy}-predictions.npy', pred)
                if split == 'test':
                    test_errors[key] = grouped
        torch.save(metadata[split], out/f'{split}-labels.pt')
    paired = {}
    for policy in policies:
        baseline = test_errors[f'slice_baseline/{policy}']
        for arm in ARMS[:-1]:
            delta = test_errors[f'{arm}/{policy}'] - baseline
            rng = np.random.default_rng(42)
            draws = [float(delta[rng.integers(len(delta), size=len(delta))].mean()) for _ in range(2000)]
            paired[f'{arm}_minus_slice_baseline/{policy}'] = dict(
                question_mean_raw_brier_difference=float(delta.mean()),
                descriptive_95_interval=np.quantile(draws, [.025, .975]).tolist(),
                questions=len(delta), scope='Paired question bootstrap; exploratory, unadjusted')
    # Within-arm comparisons distinguish epsilon effects from representation effects.
    for arm in ARMS:
        for policy in policies:
            if policy == 'legacy':
                continue
            delta = test_errors[f'{arm}/{policy}'] - test_errors[f'{arm}/legacy']
            rng = np.random.default_rng(42)
            draws = [float(delta[rng.integers(len(delta), size=len(delta))].mean()) for _ in range(2000)]
            paired[f'{arm}/{policy}_minus_legacy'] = dict(
                question_mean_raw_brier_difference=float(delta.mean()),
                descriptive_95_interval=np.quantile(draws, [.025, .975]).tolist(),
                questions=len(delta), scope='Paired question bootstrap; exploratory, unadjusted')
    # Include within-arm contrasts in the same report as representation contrasts.
    write_json(out/'summary.json', dict(protocol=protocol, fits=fits, results=results, paired=paired,
                                      seconds=time.perf_counter()-started))
    lines = [protocol['scope'], f'LayerNorm internal epsilon: {args.layernorm_eps}; Ridge epsilons are separate.',
             '', 'Test sampled-prefix Ridge: raw Brier lower; accuracy higher.',
             'arm/policy                         Brier    accuracy   epsilon      condition']
    for key, result in results['test'].items():
        lines.append(f'{key:<34} {result["raw_brier"]:.5f}  {result["accuracy"]:.3f} '
                     f'{fits[key]["epsilon"]:>11.3g} {fits[key]["condition_2"]:>12.3g}')
    lines.extend(f'{key}: {value}' for key, value in paired.items())
    (out/'report.txt').write_text('\n'.join(lines)+'\n')
    write_json(out/'status.json', dict(state='complete'))
    print('\n'.join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--strength', type=float, default=1e-4)
    parser.add_argument('--layernorm-eps', type=float, default=1e-5)
    parser.add_argument('--ridge-epsilons', type=float, nargs='+', default=[],
                        help='Additional fixed Ridge penalties; does not change LayerNorm epsilon')
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    if (not np.isfinite(args.strength) or args.strength <= 0 or not np.isfinite(args.layernorm_eps)
            or args.layernorm_eps <= 0 or args.batch_size < 1 or args.threads < 1):
        parser.error('Strength, normalization epsilon, batch size and threads must be positive')
    torch.set_num_threads(args.threads)
    run(args)


if __name__ == '__main__':
    main()
