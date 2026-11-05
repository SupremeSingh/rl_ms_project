"""Read completed local critic checkpoints on CPU; never modify a study.

Reports final systems only. Epsilon sweeps are numerical diagnostics, not
validation-selected critics or evidence of improved actor accuracy.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def spectrum(matrix):
    singular = np.linalg.svd(matrix, compute_uv=False)
    largest, smallest = float(singular[0]), float(singular[-1])
    tolerance = np.finfo(np.float64).eps * max(matrix.shape) * largest
    return dict(sigma_max=largest, sigma_min=smallest,
                condition_2=largest / smallest if smallest > 0 else None,
                numerical_rank=int(np.count_nonzero(singular > tolerance)),
                singular_values=singular.tolist())


def diagnose(a, b, epsilon):
    system = a + epsilon * np.eye(len(b))
    result = dict(epsilon=epsilon, **spectrum(system))
    try:
        w = np.linalg.solve(system, b)
        residual = np.linalg.norm(system @ w - b)
        scale = np.linalg.norm(system) * np.linalg.norm(w) + np.linalg.norm(b)
        result.update(weight_norm=float(np.linalg.norm(w)),
                      relative_residual=float(residual / max(scale, 1e-300)))
        # Fixed random RHS perturbations: a sensitivity probe, not a CI.
        rng = np.random.default_rng(42)
        amplification = []
        if np.linalg.norm(b) > 0 and np.linalg.norm(w) > 0:
            directions = rng.normal(size=(len(b), 3))
            directions /= np.linalg.norm(directions, axis=0)
            perturbations = 1e-6 * np.linalg.norm(b) * directions
            shifts = np.linalg.solve(system, perturbations)
            amplification = (np.linalg.norm(shifts, axis=0) /
                             np.linalg.norm(w) / 1e-6).tolist()
        result['relative_rhs_perturbation'] = 1e-6
        result['rhs_sensitivity_amplification'] = amplification
    except np.linalg.LinAlgError as error:
        result['solve_error'] = str(error)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--extra-epsilons', type=float, nargs='*', default=[1., 100.])
    args = parser.parse_args()
    study, output = args.study.resolve(), args.out.resolve()
    if output == study or study in output.parents:
        parser.error('Output must be outside the source study')
    if any(not np.isfinite(e) or e <= 0 for e in args.extra_epsilons):
        parser.error('Epsilons must be finite and positive')
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((study / 'manifest.json').read_text())
    report = dict(source=str(study), scope='Final saved matrices from completed runs; '
                  'different methods have different actor trajectories. Not an accuracy '
                  'comparison or a regularization recommendation.', runs=[], skipped=[])
    print('run / system                       eps        condition     min singular     |w|', flush=True)
    for name, setting in manifest['conditions'].items():
        if setting['method'] not in ('ridge', 'lstd'):
            continue
        for seed in manifest['seeds']:
            case = study / f'{name}-seed{seed}'
            if not (case / 'result.json').exists():
                report['skipped'].append(case.name)
                continue
            # Only load checkpoints produced by your own trusted local runs.
            state = torch.load(case / 'linear-critic/statistics.pt',
                               map_location='cpu', weights_only=True)
            if state['step'] != manifest['steps']:
                raise ValueError(f'Incomplete statistics: {case}')
            for index, stats in enumerate(state['stats']):
                a = stats['a'].double().numpy()
                b = stats['b'].double().numpy()
                if not np.isfinite(a).all() or not np.isfinite(b).all():
                    raise ValueError(f'Nonfinite statistics: {case}')
                record = dict(run=case.name, system=index, dimension=len(b),
                              transitions=stats['transitions'], answers=stats['answers'],
                              raw=spectrum(a), systems=[])
                for eps in sorted(set([float(state['epsilon']), *args.extra_epsilons])):
                    diagnostic = diagnose(a, b, eps)
                    diagnostic['is_training_epsilon'] = eps == float(state['epsilon'])
                    record['systems'].append(diagnostic)
                    cond = diagnostic['condition_2']
                    cond_text = f'{cond:.3e}' if cond is not None else 'inf'
                    print(f'{case.name}/{index:<2} {eps:>9g} {cond_text:>14} '
                          f'{diagnostic["sigma_min"]:>16.3e} '
                          f'{diagnostic.get("weight_norm", float("nan")):>9.3e}', flush=True)
                report['runs'].append(record)
                # Save after each system so completed diagnostic work survives a timeout.
                (output / 'conditioning.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    (output / 'conditioning.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(f'Saved {len(report["runs"])} systems to {output / "conditioning.json"}', flush=True)


if __name__ == '__main__':
    main()
