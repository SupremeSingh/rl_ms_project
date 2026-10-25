"""Linear value fitting with gamma=1 and binary terminal outcomes."""
import hashlib
import numpy as np
import torch


def question_folds(question_ids, seed):
    """Balanced, deterministic two-fold split; every answer to a question stays together."""
    ids = sorted(set(question_ids), key=lambda q: hashlib.sha256(f'{seed}:{q}'.encode()).hexdigest())
    if len(ids) < 4:
        raise ValueError('Cross-fitting needs at least four distinct questions')
    assignment = {q: i % 2 for i, q in enumerate(ids)}
    return np.array([assignment[q] for q in question_ids])


def eligibility(phi, trace_lambda, previous):
    """Parallel prefix scan of the trace recurrence; carry only within one episode."""
    if not 0 <= trace_lambda <= 1:
        raise ValueError('Trace lambda must be in [0, 1]')
    z = phi.clone()
    offset = 1
    while offset < len(z):
        z[offset:] = z[offset:] + trace_lambda ** offset * z[:-offset]
        offset *= 2
    powers = trace_lambda ** torch.arange(1, len(z) + 1, dtype=phi.dtype)
    return z + powers[:, None] * previous


def fit_system(a, b, epsilon):
    system=a.clone()
    system.diagonal().add_(epsilon)
    weights=torch.linalg.solve(system,b)
    residual=torch.linalg.vector_norm(system@weights-b)
    relative=residual/(torch.linalg.matrix_norm(system)*torch.linalg.vector_norm(weights)+torch.linalg.vector_norm(b)).clamp_min(1e-30)
    if not torch.isfinite(weights).all() or relative>1e-8:
        raise RuntimeError('Linear critic solve failed its residual check')
    return weights, float(relative)
