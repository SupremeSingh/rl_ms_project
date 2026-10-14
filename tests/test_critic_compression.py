"""Projection must preserve terminal handling, traces and raw-sum regularization."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch

from math_rl.critic_compression import affine_projection, pca_from_statistics, project_statistics
from math_rl.large_linear import statistics, accumulate, solve
from math_rl.provenance import sha256


def test_projected_statistics_match_reaccumulated_trajectories():
    torch.manual_seed(9)
    states = [torch.randn(n, 5, dtype=torch.float64) + 3 for n in (7, 11, 1)]
    rewards = [1, 0, 1]
    original = statistics(5)
    mean, scale = torch.randn(5).double(), torch.full((5,), 2.).double()
    for x, r in zip(states, rewards):
        accumulate(original, x, r, mean, scale, chunk=3)
    center, spectrum, vectors = pca_from_statistics(original)
    all_x = torch.cat([(x-mean)/scale for x in states])
    torch.testing.assert_close(center, all_x.mean(0))
    torch.testing.assert_close(spectrum.sum(), all_x.var(0, unbiased=False).sum())
    transform = affine_projection(center, vectors[:, :3])
    projected = project_statistics(original, transform)
    direct = statistics(3)
    for x, r in zip(states, rewards):
        compressed = ((x-mean)/scale-center) @ vectors[:, :3]
        accumulate(direct, compressed, r, torch.zeros(3), torch.ones(3), chunk=2)
    for k in direct:
        if torch.is_tensor(direct[k]):
            torch.testing.assert_close(direct[k], projected[k])
        else:
            assert direct[k] == projected[k]
    for head in ('ridge', 'lstd-0.99'):
        torch.testing.assert_close(solve(direct, head)[0], solve(projected, head)[0])
    # A zero terminal state must not acquire a centered offset or intercept.
    assert torch.equal(torch.zeros(6, dtype=torch.float64) @ transform, torch.zeros(4, dtype=torch.float64))


def test_complete_pipeline_and_resume(tmp_path):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root/'scripts'))
    spec = importlib.util.spec_from_file_location('compression_runner', root/'scripts/compress_jlens_critics.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source, out = tmp_path/'source', tmp_path/'result'
    source.mkdir()
    (source/'manifest.json').write_text(json.dumps({'config': {'shards': 1,
        'linear_gamma': 1., 'linear_lambda': .99, 'linear_epsilon': .01}}))
    (source/'linear-statistics').mkdir()
    torch.manual_seed(31)
    stats, normalization = {}, {}
    labels = dict(y=torch.tensor([0., 1.]*8), question=torch.arange(16)//2,
                  response=torch.arange(16), position=torch.zeros(16, dtype=torch.long))
    for arm in module.ARMS + module.BASELINES:
        x = torch.randn(16, 5)
        stats[arm] = statistics(5)
        for i in range(16):
            accumulate(stats[arm], x[i:i+1], int(labels['y'][i]), torch.zeros(5), torch.ones(5))
        normalization[arm] = (torch.zeros(5), torch.ones(5))
        dest = source/'packed'/arm
        dest.mkdir(parents=True)
        files = {}
        for split in ('val', 'test'):
            np.save(dest/f'{split}-x.npy', x.numpy())
            torch.save(labels, dest/f'{split}-labels.pt')
            for suffix in ('x.npy', 'labels.pt'):
                p = dest/f'{split}-{suffix}'
                files[p.name] = sha256(p)
        (dest/'complete.json').write_text(json.dumps(dict(files=files)))
    checkpoint = source/'linear-statistics/shard-0.pt'
    torch.save(dict(stats=stats, normalization=normalization), checkpoint)
    checkpoint.with_suffix('.json').write_text(json.dumps(dict(sha256=sha256(checkpoint))))
    args = SimpleNamespace(source=source, out=out, dimensions=[2, 3])
    module.run(args)
    report = json.loads((out/'summary.json').read_text())
    assert len(report['results']) == 22
    assert len(report['selection']) == 10
    module.run(args)
    assert json.loads((out/'summary.json').read_text()) == report
