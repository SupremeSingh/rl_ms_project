"""Opt-in recovery of sealed encoder studies; never modifies the source study.

Preparation and status use only the standard library. No jobs are submitted.
"""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def checked(path):
    path = Path(path)
    if digest(path) != json.loads(path.with_suffix(path.suffix + '.sha256').read_text())['sha256']:
        raise ValueError(f'Changed artifact: {path}')
    return json.loads(path.read_text())


def combinations(m):
    return [(b, a, s) for b in ['causal'] + [f'adapted-{s}' for s in m['adapt_seeds']]
            for a in 'ABCD' for s in (m['compressor_seeds'] if a in 'CD' else [0])]


def validation_report(source, out):
    """Read only sealed completed A/B fits; no feature loading or new fitting."""
    print('Loading PyTorch to read completed small checkpoints...', flush=True)
    import torch
    torch.set_num_threads(1)
    source, out = source.resolve(), out.resolve()
    if source == out or source in out.parents or out in source.parents:
        raise ValueError('Report output must be separate from source')
    m = checked(source/'manifest.json')
    results = {}
    for b in ['causal'] + [f'adapted-{s}' for s in m['adapt_seeds']]:
        results[b] = {}
        for arm in 'AB':
            folder = source/'fits'/f'{b}-{arm}-0'
            if not (folder/'complete.json').exists():
                continue
            if digest(folder/'model.pt') != checked(folder/'complete.json')['sha256']:
                raise ValueError(f'Changed checkpoint: {folder}')
            r = torch.load(folder/'model.pt', map_location='cpu', weights_only=True)
            results[b][arm] = float(r['validation_brier'])
    lines = ['PAUSED STUDY: completed direct-readout validation results.',
             'Lower raw Brier is better; no test or MC evaluation. C/D unfinished.',
             'A: final-layer mean. B: two-thirds-layer mean.']
    for b, values in results.items():
        lines.append(f'{b}: {values}')
        if b != 'causal':
            for arm in values.keys() & results['causal'].keys():
                lines.append(f'  {arm} encoder-minus-causal: {values[arm]-results["causal"][arm]:+.8f}')
    out.mkdir(parents=True, exist_ok=False)
    (out/'summary.json').write_text(json.dumps(dict(source=str(source),
        manifest_sha256=digest(source/'manifest.json'), validation_brier=results,
        scope=lines[:3]), indent=2)+'\n')
    (out/'report.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines), flush=True)
    print('Saved:', out/'report.txt', flush=True)


def prepare(source, out):
    source, out = source.resolve(), out.resolve()
    if source == out or source in out.parents or out in source.parents:
        raise ValueError('Recovery output must be separate from source')
    m = checked(source / 'manifest.json')
    checked(source / 'questions.json')
    for b in ['causal'] + [f'adapted-{s}' for s in m['adapt_seeds']]:
        for shard in range(m['shards']):
            if not checked(source / f'extract-{b}-{shard}.json')['passed']:
                raise ValueError('Extraction is incomplete')
    reusable, missing = [], []
    for task, (b, a, seed) in enumerate(combinations(m)):
        name = f'{b}-{a}-{seed}'
        folder = source / 'fits' / name
        if (folder / 'complete.json').exists():
            if digest(folder / 'model.pt') != checked(folder / 'complete.json')['sha256']:
                raise ValueError(f'Changed fit: {name}')
            reusable.append(name)
        else:
            missing.append(task)
    # Validate everything above before creating the fresh recovery directory.
    out.mkdir(parents=True, exist_ok=False)
    for name in ('manifest.json', 'manifest.json.sha256', 'questions.json',
                 'questions.json.sha256', 'answers', 'features'):
        p = source / name
        (out / name).symlink_to(p, target_is_directory=p.is_dir())
    (out / 'fits').mkdir()
    for name in reusable:
        (out / 'fits' / name).symlink_to(source / 'fits' / name, target_is_directory=True)
    mc_missing = []
    for shard in range(m['shards']):
        name = f'mc-{shard}.json'
        if (source / name).exists():
            refs = checked(source / name)
            # The final evaluator consumes the sealed aggregate, not raw batches.
            if not isinstance(refs, list):
                raise ValueError(f'Invalid MC aggregate: {name}')
            for suffix in ('', '.sha256'):
                (out / (name + suffix)).symlink_to(source / (name + suffix))
        else:
            mc_missing.append(shard)
    plan = dict(source=str(source), manifest_sha256=digest(source / 'manifest.json'),
                original_runtime=m['runtime'], recovery_script_sha256=digest(__file__),
                reused_fits=reusable, missing_fit_tasks=missing, missing_mc_tasks=mc_missing,
                scope='Numerical/memory recovery; original splits, samples, losses and budgets retained. '
                      'Normalization uses bounded blocks; floating-point roundoff may differ.')
    (out / 'recovery.json').write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps(plan, indent=2))


def bounded_normalization(x, batch=2048):
    """Same centered scalar RMS as SLICE, with O(batch * width) temporaries."""
    import torch
    if len(x) == 0:
        raise ValueError('No training features')
    total = torch.zeros(x.shape[1], dtype=torch.float64)
    for start in range(0, len(x), batch):
        total += x[start:start + batch].double().sum(0)
    mean = (total / len(x)).float()
    squares = torch.zeros((), dtype=torch.float64)
    for start in range(0, len(x), batch):
        # Center in float32, matching the original operation order.
        block = (x[start:start + batch] - mean).double()
        squares += block.square().sum()
    scale = (squares / x.numel()).sqrt().clamp_min(1e-8).float()
    return mean, scale.expand_as(mean).clone()


def cache_training(c, out, m, backbone, arm, cache):
    """Disk-backed sampled matrix: no list retaining all per-answer tensors."""
    import numpy as np
    import shutil
    import torch
    rows = [r for r in c.items(out, m) if r['split'] == 'train']
    width = len(c.LAYERS) * 1536
    capacity = len(rows) * 8
    required = capacity * (width + 1) * 4
    if not capacity or shutil.disk_usage(cache).free < required + 1024**3:
        raise ValueError(f'Need nonempty training data and at least {required/1024**3+1:.1f} GiB scratch space')
    matrix = np.memmap(cache / 'x.f32', mode='w+', dtype='float32', shape=(capacity, width))
    labels = np.memmap(cache / 'y.f32', mode='w+', dtype='float32', shape=(capacity,))
    offset = 0
    for i, row in enumerate(rows):
        r = c.feature(out, backbone, row)
        g = torch.Generator().manual_seed(c.stable_seed('prefixes', row['key']))
        ids = torch.randperm(len(r['y']), generator=g)[:8]
        n = len(ids)
        matrix[offset:offset+n] = c.xvalue(r, arm)[ids].numpy()
        labels[offset:offset+n] = r['y'][ids].numpy()
        offset += n
        if (i+1) % 5000 == 0:
            print(f'Cached {i+1}/{len(rows)} train answers; {offset} prefixes', flush=True)
    matrix.flush(); labels.flush()
    x, y = torch.from_numpy(matrix[:offset]), torch.from_numpy(labels[:offset])
    if set(y.tolist()) != {0., 1.}:
        raise ValueError('Need both outcome classes in fitting data')
    return x, y


def fit(c, out, m, task):
    import tempfile
    import torch
    b, a, seed = c.combinations(m)[task]
    folder = out / 'fits' / f'{b}-{a}-{seed}'
    if (folder / 'complete.json').exists():
        if digest(folder/'model.pt') != checked(folder/'complete.json')['sha256']:
            raise ValueError('Changed completed fit')
        print('Reusing completed fit:', folder, flush=True)
        return
    if a in 'AB':
        c.fit(out, m, task)
        return
    folder.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    # Prefer scheduler-local scratch. Scratch is unique per fit and cleaned on success/error.
    with tempfile.TemporaryDirectory(prefix='slice-fit-') as cache:
        x, y = cache_training(c, out, m, b, a, Path(cache))
        mean, scale = bounded_normalization(x)
        width = 1536 if a == 'C' else 768
        model = c.Bottleneck(x.shape[1], width).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.001)
        best, history = float('inf'), []
        for epoch in range(1, m['epochs']+1):
            model.train()
            for ids in torch.randperm(len(x)).split(2048):
                objective = c.loss(model, ((x[ids]-mean)/scale).cuda(), y[ids].cuda(), .01)
                if not torch.isfinite(objective):
                    raise ValueError('Nonfinite compressor loss')
                opt.zero_grad(); objective.backward(); opt.step()
            if epoch % m['checkpoint_every'] and epoch != m['epochs']:
                continue
            model.eval()
            head = c.fit_head(out, m, b, a, model, mean, scale)
            candidate = dict(head, compressor={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                             mean=mean, scale=scale, width=width, epoch=epoch)
            val = c.score(out, m, b, a, candidate, 'val')[0]
            history.append(dict(epoch=epoch, validation_brier=val))
            print(b, a, seed, history[-1], flush=True)
            if val < best:
                best, record = val, candidate
        record.update(validation_brier=best, history=history)
        torch.save(record, folder/'model.pt')
        c.seal(out, f'fits/{folder.name}/complete.json', dict(sha256=digest(folder/'model.pt')))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['prepare', 'validation', 'status', 'fit', 'mc', 'evaluate'], required=True)
    p.add_argument('--study', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--task', type=int, default=0)
    args = p.parse_args()
    out = args.out.resolve()
    if args.stage in ('prepare', 'validation'):
        if args.study is None:
            p.error('--study is required for prepare/validation')
        (prepare if args.stage == 'prepare' else validation_report)(args.study, out)
        return
    plan = json.loads((out/'recovery.json').read_text())
    if args.stage == 'status':
        m = checked(out/'manifest.json')
        missing = [i for i, (b,a,s) in enumerate(combinations(m))
                   if not (out/'fits'/f'{b}-{a}-{s}'/'complete.json').exists()]
        print(json.dumps(dict(missing_fit_tasks=missing,
                              missing_mc_tasks=[i for i in range(m['shards'])
                                                if not (out/f'mc-{i}.json').exists()]), indent=2))
        return
    import compare as c
    c.torch.set_num_threads(4)
    if digest(__file__) != plan['recovery_script_sha256']:
        raise ValueError('Recovery script changed after preparation')
    source = Path(plan['source'])
    if digest(source/'manifest.json') != plan['manifest_sha256']:
        raise ValueError('Source manifest changed')
    m = c.validate(source)
    if args.stage == 'fit':
        if not 0 <= args.task < len(c.combinations(m)):
            p.error('Invalid fitting task')
        fit(c, out, m, args.task)
    elif args.stage == 'mc':
        if not 0 <= args.task < m['shards']:
            p.error('Invalid MC task')
        c.mc(out, m, args.task)
    else:
        c.evaluate(out, m)


if __name__ == '__main__':
    main()
