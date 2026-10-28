"""Verify trainer checkpoints, accumulation and one/two-system contracts."""
import json
from pathlib import Path


def verify_run(case, method, *, folds=2, steps=2, evaluation_split="val", feature_width=768):
    if type(folds) is not int or folds not in (1, 2):
        raise ValueError('Expected one or two critic systems')
    dimension = feature_width + 1
    case = Path(case)
    import torch
    result = json.loads((case / 'result.json').read_text())
    if result['updates'] != steps or result['answers'] != steps * 64 or result['evaluation_split'] != evaluation_split:
        raise ValueError('Incomplete actor update/evaluation budget')
    stats = torch.load(case / 'linear-critic' / 'statistics.pt', map_location='cpu', weights_only=True)
    if (stats['step'] != steps or stats['epoch'] != 0 or stats['method'] != method
            or stats['epsilon'] != .01 or stats['trace_lambda'] != .9):
        raise ValueError('Wrong cumulative fitting settings')
    if len(stats['stats']) != folds or stats.get('folds', 2) != folds:
        raise ValueError('Wrong number of critic systems')
    if folds == 1 and (not stats.get('assignment') or set(stats['assignment'].values()) != {0}):
        raise ValueError('No-fold critic must assign every question to the shared system')
    if sum(s['answers'] for s in stats['stats']) != steps * 64:
        raise ValueError('Statistics did not retain all updates')
    for s in stats['stats']:
        if (s['a'].shape != (dimension, dimension) or s['b'].shape != (dimension,) or not s['answers']
                or not torch.isfinite(s['a']).all() or not torch.isfinite(s['b']).all()):
            raise ValueError('Invalid critic matrices')
    metrics = [json.loads(row) for row in (case / 'linear-critic' / 'metrics.jsonl').read_text().splitlines()]
    if len(metrics) != steps or any(m['reset'] != (i == 0) or m['feature_epoch'] != 0 for i,m in enumerate(metrics)):
        raise ValueError('Feature basis or matrices unexpectedly reset')
    if folds == 1 and any(m.get('critic_folds') != 1 or m.get('cross_fitted') is not False for m in metrics):
        raise ValueError('Missing or inconsistent no-fold metadata')
    if any(m.get('critic_folds', 2) != folds or m.get('cross_fitted', True) != (folds == 2)
           for m in metrics):
        raise ValueError('Inconsistent fold metadata')
    for step in range(1, steps + 1):
        record = torch.load(case / 'linear-critic' / f'{step}.pt', map_location='cpu', weights_only=True)
        if len(record['heads']) != folds or any(h['weights'].shape != (dimension,)
                or not torch.isfinite(h['weights']).all() for h in record['heads']):
            raise ValueError('Invalid fresh critic heads')
        for predicted_fold, head in enumerate(record['heads']):
            fitting_fold = 0 if folds == 1 else 1 - predicted_fold
            if (head.get('predicted_fold') != predicted_fold or head.get('fitting_fold') != fitting_fold
                    or head.get('feature_epoch') != 0):
                raise ValueError('Head uses the wrong fitting/prediction system')
            if step == steps:
                system = stats['stats'][fitting_fold]
                a = system['a'].double() + .01 * torch.eye(dimension, dtype=torch.float64)
                b = system['b'].double()
                weights = head['weights'].double()
                residual = torch.linalg.vector_norm(a @ weights - b)
                scale = (torch.linalg.matrix_norm(a) * torch.linalg.vector_norm(weights)
                         + torch.linalg.vector_norm(b)).clamp_min(1e-30)
                if residual / scale > 1e-8:
                    raise ValueError('Saved head does not solve its accumulated system')
    return dict(passed=True, updates=steps, answers=steps*64, matrix_dimension=dimension,
                folds=folds, cross_fitted=folds == 2, statistics_answers=[s['answers'] for s in stats['stats']],
                training_gpu_hours=result['training_gpu_hours'])

