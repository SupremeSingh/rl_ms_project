"""Saved Numina feature arrays and statistics; compatible with existing study files."""
import json
import numpy as np
import torch
import frozen_critics as frozen
from math_rl.provenance import sha256


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    frozen.atomic_json(path, value)


def trajectory(out, q):
    return out/'trajectories'/f'{q["index"]:05d}.json'


def dataset(out, arm, split):
    dest = out/'packed'/arm
    record = read(dest/'complete.json')
    for suffix in ('x.npy', 'labels.pt'):
        path = dest/f'{split}-{suffix}'
        if sha256(path) != record['files'][path.name]:
            raise ValueError('Packed data changed')
    data = torch.load(dest/f'{split}-labels.pt', weights_only=True)
    # Copy-on-write mmap: tensor indexing copies only minibatches, not the whole feature matrix.
    data['x'] = torch.from_numpy(np.load(dest/f'{split}-x.npy', mmap_mode='c'))
    return data


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


def source_feature_environment(source):
    """Read neutral runtime metadata, with compatibility for historical studies."""
    path = source/'feature-runtime.json'
    if not path.exists():
        # This is a saved filename only; no lens package is imported or executed.
        path = source/'lens-metadata.json'
    return read(path)['environment']
