"""Separate, validation-only causal versus one-adaptation-seed analysis.

Reuses compare.fit unchanged; redirects its input readers to the sealed study.
Never writes into the source study or loads test feature tensors.
"""
import argparse
import json
from pathlib import Path
import statistics


def configure(c, study, out, seed):
    study, out = study.resolve(), out.resolve()
    if out == study or study in out.parents or out in study.parents:
        raise ValueError('Interim output must be separate from the source study')
    m = c.validate(study)
    if seed not in m['adapt_seeds']:
        raise ValueError('Requested adaptation seed is absent')
    for b in ('causal', f'adapted-{seed}'):
        for shard in range(m['shards']):
            if not c.checked(study, f'extract-{b}-{shard}.json')['passed']:
                raise ValueError(f'Incomplete extraction: {b}/{shard}')
    rows = [r for r in c.items(study, m) if r['split'] in ('train', 'val')]
    allowed = {r['key'] for r in rows}
    source_feature = c.feature

    def feature(_out, b, row):
        if row['key'] not in allowed:
            raise ValueError('Interim analysis cannot read test features')
        return source_feature(study, b, row)

    c.items = lambda _out, _m: rows
    c.feature = feature
    m = dict(m, adapt_seeds=[seed])
    out.mkdir(parents=True, exist_ok=True)
    identity = dict(study=str(study), manifest_sha256=c.sha256(study/'manifest.json'),
                    seed=seed, scope='Exploratory validation only; no test or MC evaluation',
                    runtime=c.identity(), interim_script_sha256=c.sha256(Path(__file__)))
    return m, identity


def report(c, out, m, identity):
    import torch
    values = {}
    for task, (b, a, seed) in enumerate(c.combinations(m)):
        root = out/f'task-{task}'
        if json.loads((root/'source.json').read_text()) != identity:
            raise ValueError('Interim source or runtime changed')
        path = root/'fits'/f'{b}-{a}-{seed}'/'model.pt'
        meta = c.checked(root, f'fits/{b}-{a}-{seed}/complete.json')
        if c.sha256(path) != meta['sha256']:
            raise ValueError('Fit checksum mismatch')
        record = torch.load(path, map_location='cpu', weights_only=True)
        values.setdefault(f'{b}/{a}', []).append(float(record['validation_brier']))
    summary = {k: dict(mean=statistics.mean(v), by_compressor_seed=v) for k,v in values.items()}
    lines = ['PRELIMINARY validation comparison; lower raw Brier is better.',
             'One encoder adaptation seed. Validation also selects compressor checkpoints.',
             'No test evaluation, MC-reference accuracy, or significance claim.']
    seed = m['adapt_seeds'][0]
    for a in c.ARMS:
        baseline = summary[f'causal/{a}']['mean']
        encoder = summary[f'adapted-{seed}/{a}']['mean']
        lines.append(f'{a}: causal={baseline:.6f}; encoder={encoder:.6f}; encoder-causal={encoder-baseline:+.6f}')
    (out/'summary.json').write_text(json.dumps(summary, indent=2))
    (out/'report.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--stage', choices=['fit', 'report'], required=True)
    p.add_argument('--task', type=int, default=0)
    args = p.parse_args()
    import compare as c
    c.torch.set_num_threads(4)
    m, identity = configure(c, args.study, args.out, args.seed)
    out = args.out.resolve()
    if args.stage == 'fit':
        if not 0 <= args.task < len(c.combinations(m)):
            p.error('Invalid task index')
        root = out/f'task-{args.task}'
        root.mkdir(exist_ok=True)
        with (root/'source.json').open('x') as f:
            json.dump(identity, f, indent=2)
        c.fit(root, m, args.task)
    else:
        report(c, out, m, identity)


if __name__ == '__main__':
    main()
