"""Retrospective head-snapshot MAE against FIXED final-policy MC references.

Uses the earlier completed value-repeat audit, not the new learning-curve actors.
No sampling, training, refitting, or source writes. Not an on-policy MAE curve.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean, stdev
import sys
import time


def question_mae(states, predictions, references):
    grouped = defaultdict(list)
    for state, prediction in zip(states, predictions, strict=True):
        grouped[state['question_id']].append(abs(float(prediction) - references[state['state_id']]))
    return mean(mean(errors) for errors in grouped.values())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[1101, 1102, 1103])
    p.add_argument('--steps', type=int, nargs='+', default=[20, 40, 60, 80, 100, 120])
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[2]
    # Also permit copying this standalone script into the cluster's outputs/.
    if not (repo / 'src/math_rl').exists():
        repo = Path.cwd().resolve()
    sys.path.insert(0, str(repo / 'src'))
    import numpy as np
    import torch
    from transformers import AutoModel
    from slice_critic.online import bottleneck_prefixes, load_filter
    from slice_critic.storage import sha256, model_identity, write_json
    from math_rl.value_audit_gpu import load_tokenizer, tokenizer_identity

    source, out = a.source.resolve(), a.out.resolve()
    if out == source or source in out.parents or out.exists():
        raise ValueError('Require a NEW output directory outside the source study')
    if not a.seeds or len(set(a.seeds)) != len(a.seeds):
        raise ValueError('Require distinct seeds')
    if not a.steps or a.steps != sorted(set(a.steps)) or a.steps[0] < 1 or a.steps[-1] != 120:
        raise ValueError('Use increasing head steps ending at 120')
    read = lambda path: json.loads(path.read_text())
    parent = read(source / 'manifest.json')
    v = read(source / 'main/verification.json')
    if not v['passed'] or v['manifest_sha256'] != sha256(source / 'manifest.json'):
        raise ValueError('Source study verification mismatch')
    if not set(a.seeds) <= set(parent['seeds']):
        raise ValueError('Unknown source seeds')
    basis = model_identity(repo / 'models/qwen-math')
    arms = ['slice-token', 'slice-chunk']
    prepared = []
    hashes = {}
    for seed in a.seeds:
        child = source / f'seed{seed}'
        m = read(child / 'manifest.json')
        if sha256(child / 'manifest.json') != parent['children'][str(seed)] or m['model_hashes'] != basis:
            raise ValueError('Changed source manifest/encoder')
        folder = child / 'full'
        bank_path = folder / 'bank.json'
        bank = read(bank_path)
        states = bank['states']
        ids = {s['state_id'] for s in states}
        if len(ids) != len(states):
            raise ValueError('Duplicate bank states')
        identity = dict(manifest_sha256=sha256(child / 'manifest.json'), bank_sha256=sha256(bank_path))
        child_v = read(folder / 'verification.json')
        if sha256(folder / 'verification.json') != v['child_verification_sha256'][str(seed)]:
            raise ValueError('Child verification differs from completed aggregate')
        if not child_v['passed'] or any(child_v[k] != h for k, h in identity.items()):
            raise ValueError('Unverified source bank')
        hashes[str(bank_path)] = sha256(bank_path)
        if sha256(child / 'filters/sparse.pt') != m['filter_sha256']:
            raise ValueError('Changed compressor')
        references, heads, final_predictions = {}, {}, {}
        for arm in arms:
            refs = {}
            for shard in range(m['budgets']['full']['shards']):
                path = folder / f'reference-{arm}-{shard}.json'
                r = read(path)
                if any(r[k] != h for k, h in identity.items()):
                    raise ValueError('Reference identity mismatch')
                hashes[str(path)] = sha256(path)
                for row in r['states']:
                    if row['state_id'] in refs or row['samples'] != 128 or not 0 <= row['successes'] <= 128:
                        raise ValueError('Incomplete or duplicate reference')
                    refs[row['state_id']] = row['successes'] / 128
            if set(refs) != ids:
                raise ValueError('Reference state coverage differs')
            references[arm] = refs
            case = child / f'{arm}-seed{seed}'
            if read(case / 'result.json')['updates'] != 120:
                raise ValueError('Incomplete original run')
            heads[arm] = {}
            for step in a.steps:
                path = case / f'linear-critic/{step}.pt'
                record = torch.load(path, map_location='cpu', weights_only=True)
                if record['step'] != step or record['method'] != 'ridge' or len(record['heads']) != 1 or record['feature_policy'] != 'frozen':
                    raise ValueError('Unsupported head snapshot')
                heads[arm][step] = record['heads'][0]['weights'].double().reshape(-1)
                if record['heads'][0]['feature_epoch'] != 0 or heads[arm][step].numel() != 769:
                    raise ValueError('Feature basis changed')
                hashes[str(path)] = sha256(path)
            path = folder / f'predictions-{arm}.json'
            r = read(path)
            if any(r[k] != h for k, h in identity.items()):
                raise ValueError('Final prediction identity mismatch')
            final_predictions[arm] = {s['state_id']: s['prediction'] for s in r['states']}
            hashes[str(path)] = sha256(path)
            for name in [bank_path.name, path.name] + [f'reference-{arm}-{j}.json' for j in range(m['budgets']['full']['shards'])]:
                if child_v['files'][name] != sha256(folder / name):
                    raise ValueError('Original verified artifact changed')
        actor = child / f'slice-token-seed{seed}/global_step_120/actor/huggingface'
        tokenizer = load_tokenizer(actor)
        if tokenizer_identity(tokenizer) != bank['tokenizer_sha256']:
            raise ValueError('Tokenizer differs from sampled actions')
        prepared.append((seed, child, m, states, references, heads, final_predictions, tokenizer.eos_token_id))

    if not torch.cuda.is_available():
        raise RuntimeError('One GPU is required for the frozen-feature pass')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    started = time.perf_counter()
    model = AutoModel.from_pretrained(repo / 'models/qwen-math', local_files_only=True,
        trust_remote_code=False, torch_dtype=torch.float32, attn_implementation='eager').to('cuda').eval().requires_grad_(False)
    out.mkdir(parents=True, exist_ok=False)
    results = {arm: {} for arm in arms}
    for seed, child, m, states, refs, heads, final, eos in prepared:
        fitted = load_filter(child / 'filters/sparse.pt', m['filter_sha256'], layers=m['layers'],
                             eps=m['rms_eps'], model_path=repo / 'models/qwen-math')
        features = []
        with torch.inference_mode():
            for i, state in enumerate(states):
                ids = torch.tensor([state['prompt_ids'] + state['prefix_ids'] + [eos]])
                x = torch.as_tensor(bottleneck_prefixes(model, ids, torch.ones_like(ids), 1,
                                       fitted, m['layers'], m['rms_eps'])[0]).cpu().double()
                features.append(torch.cat([x, torch.ones(1, 1, dtype=torch.float64)], 1).reshape(-1))
                if (i + 1) % 25 == 0:
                    print(f'Seed {seed}: {i+1}/{len(states)} shared features', flush=True)
        design = torch.stack(features)
        torch.save(dict(state_ids=[s['state_id'] for s in states], design=design,
                        bank_sha256=sha256(child / 'full/bank.json')), out / f'features-seed{seed}.pt')
        for arm in arms:
            rows = []
            for step in a.steps:
                pred = (design @ heads[arm][step]).numpy()
                if not np.isfinite(pred).all():
                    raise ValueError('Nonfinite predictions')
                if step == 120:
                    delta = max(abs(float(p) - final[arm][s['state_id']]) for s, p in zip(states, pred))
                    if delta > 1e-4:
                        raise ValueError(f'Final-head predictions fail original-audit reproduction: {delta}')
                rows.append(dict(step=step, mae=question_mae(states, pred, refs[arm]),
                                 predictions=list(map(float, pred))))
            results[arm][str(seed)] = rows
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.7))
    for arm, color in zip(arms, ['#2765A7', '#A064AA']):
        means = [mean(results[arm][str(s)][i]['mae'] for s in a.seeds) for i in range(len(a.steps))]
        ax.plot(a.steps, means, 'o-', color=color, label=arm.replace('slice', 'SLICE'))
        if len(a.seeds) > 1:
            sd = [stdev(results[arm][str(s)][i]['mae'] for s in a.seeds) for i in range(len(a.steps))]
            ax.fill_between(a.steps, np.array(means)-sd, np.array(means)+sd, color=color, alpha=.13)
        print(f'{arm}: {dict(zip(a.steps, means))}', flush=True)
    ax.set(xlabel='Saved Ridge-head update', ylabel='MAE vs final-policy MC reference',
           title=f'Fixed final-policy target · {len(a.seeds)} seed(s) · 128 MC samples/state')
    ax.grid(alpha=.2); ax.legend()
    fig.text(.5, .025, 'Retrospective diagnostic; NOT each checkpoint’s own-policy value error.\nSame prefixes/references at all steps; token/chunk retain different final actors.',
             ha='center', fontsize=9, color='#555555')
    fig.subplots_adjust(bottom=.23)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(out / f'fixed-target-head-mae.{ext}', dpi=220, bbox_inches='tight')
    plt.close(fig)
    write_json(out / 'summary.json', dict(results=results, seeds=a.seeds, steps=a.steps,
        source=str(source), source_hashes=hashes, elapsed_seconds=time.perf_counter()-started,
        scope='Earlier value-repeat study. Fixed final-policy MC references; no new MC, fitting or training. Equal-question MAE; raw predictions. Final head precedes final actor by one update. Not a critic-learning on-policy error curve.'))
    write_json(out / 'verification.json', dict(passed=True, files={name:sha256(out / name)
        for name in ['summary.json', 'fixed-target-head-mae.png', 'fixed-target-head-mae.pdf']}))
    print(f'Saved {out}/fixed-target-head-mae.png', flush=True)


if __name__ == '__main__':
    main()
