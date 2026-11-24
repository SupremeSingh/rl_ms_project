"""Replay or recompute a saved five-critic, fixed-policy offline comparison.

Uses existing artifacts only. Never generates rollouts, trains models, or writes
into the source study. The transformer here is MC-fitted, not a PPO run.
"""
import argparse
import csv
import gc
import math
from pathlib import Path
from statistics import mean

from .storage import read_json, sha256, write_json

ARMS = ('slice-token', 'slice-chunk', 'rms-layer-token', 'rms-layer-chunk', 'transformer')
TITLES = ('SLICE/token', 'SLICE/chunk', 'RMS layer/token', 'RMS layer/chunk',
          'Full transformer critic (MC fit)')


def indexed(rows):
    result = {}
    for row in rows:
        sid = row['state_id']
        if sid in result:
            raise ValueError(f'Duplicate state: {sid}')
        result[sid] = row
    return result


def load_study(folder):
    folder = Path(folder)
    manifest = read_json(folder / 'manifest.json')
    if manifest.get('protocol') != 'fixed-policy-critics-v1':
        raise ValueError('Expected the fixed-policy-critics-v1 study, not a multi-actor PPO audit')
    identity = dict(manifest_sha256=sha256(folder / 'manifest.json'),
                    bank_sha256=sha256(folder / 'bank.json'))
    def check(record, bank=True):
        keys = identity if bank else {'manifest_sha256': identity['manifest_sha256']}
        if any(record.get(k) != v for k, v in keys.items()):
            raise ValueError('Study identity mismatch')
    bank = read_json(folder / 'bank.json')
    check(bank, bank=False)
    states = indexed(bank['states'])
    if not states:
        raise ValueError('Empty evaluation bank')
    refs = []
    for shard in range(manifest['shards']):
        record = read_json(folder / f'reference-{shard}.json')
        check(record)
        refs.extend(record['states'])
    references = indexed(refs)
    if set(states) != set(references):
        raise ValueError('Incomplete or unexpected reference states')
    for r in references.values():
        if (type(r['samples']) is not int or r['samples'] != manifest['recipe']['samples']
                or type(r['successes']) is not int or not 0 <= r['successes'] <= r['samples']):
            raise ValueError('Invalid MC reference counts')
    return manifest, identity, states, references, check


def replay(folder, states, check):
    predictions = {}
    for arm in ARMS:
        record = read_json(Path(folder) / f'predictions-{arm}.json')
        check(record)
        rows = indexed(record['states'])
        if set(rows) != set(states):
            raise ValueError(f'{arm}: incomplete or unexpected predictions')
        predictions[arm] = {sid: float(r['prediction']) for sid, r in rows.items()}
    return predictions


def recompute(folder, manifest, states, encoder_path, device):
    import torch
    from .backend import TransformersBackend
    from .inference import load_filter
    from .layers import layer_spec
    from .storage import model_identity

    folder, encoder_path = Path(folder), Path(encoder_path)
    if model_identity(encoder_path) != manifest['base_identity']:
        raise ValueError('Frozen encoder differs from the saved study')
    linear = read_json(folder / 'linear.json')
    if (linear['manifest_sha256'] != sha256(folder / 'manifest.json')
            or linear['sha256'] != sha256(folder / 'ridge.pt')):
        raise ValueError('Ridge checkpoint identity mismatch')
    heads = torch.load(folder / 'ridge.pt', map_location='cpu', weights_only=True)
    compressor, center, scale = load_filter(folder / 'sparse.pt', manifest['filter_sha256'],
        layers=manifest['layers'], eps=manifest['rms_eps'], model_path=encoder_path)
    backend = TransformersBackend(encoder_path, device=device)
    raw_layer = layer_spec(backend.num_layers)['two_thirds'] - 1
    layers = list(dict.fromkeys(manifest['layers'] + [raw_layer]))
    predictions = {arm: {} for arm in ARMS}
    with torch.inference_mode():
        for sid, state in states.items():
            features = backend.state_features(state['prompt_ids'] + state['prefix_ids'], layers)
            blocks = dict(zip(layers, features.split(backend.hidden_size, dim=1), strict=True))
            x = torch.cat([blocks[layer] for layer in manifest['layers']], dim=1)
            z = compressor.features((x - center) / scale)
            for arm in ARMS[:-1]:
                f = z if arm.startswith('slice') else blocks[raw_layer]
                design = torch.cat([f.double(), torch.ones(1, 1, dtype=torch.float64)], dim=1)
                w = heads[arm]['weights'].double().cpu()
                if w.shape != (design.shape[1],):
                    raise ValueError(f'{arm}: feature/head dimensions differ')
                predictions[arm][sid] = float((design @ w).item())
    del backend, compressor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    neural = read_json(folder / 'neural.json')
    if (neural['manifest_sha256'] != sha256(folder / 'manifest.json')
            or model_identity(folder / 'transformer') != neural['model_identity']):
        raise ValueError('Transformer checkpoint identity mismatch')
    from transformers import AutoModelForTokenClassification
    model, info = AutoModelForTokenClassification.from_pretrained(folder / 'transformer',
        local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16 if str(device).startswith('cuda') else torch.float32,
        attn_implementation='sdpa', output_loading_info=True)
    if (model.config.num_labels != 1 or info['missing_keys'] or info['unexpected_keys']
            or info.get('mismatched_keys')):
        raise ValueError('Expected a complete scalar transformer critic')
    model.to(device).eval().requires_grad_(False)
    with torch.inference_mode():
        for sid, state in states.items():
            ids = torch.tensor([state['prompt_ids'] + state['prefix_ids']], device=device)
            if not ids.numel() or ids.shape[1] > model.config.max_position_embeddings:
                raise ValueError('Empty input or context limit exceeded')
            predictions['transformer'][sid] = float(model(input_ids=ids,
                attention_mask=torch.ones_like(ids), use_cache=False).logits[0, -1, 0])
    return predictions


def summarize(states, references, predictions):
    groups = {}
    for sid, s in states.items():
        groups.setdefault(s['question_id'], []).append(sid)
    result = {}
    for arm in ARMS:
        if any(not math.isfinite(v) for v in predictions[arm].values()):
            raise ValueError(f'{arm}: nonfinite predictions')
        errors = {sid: predictions[arm][sid] - r['successes'] / r['samples']
                  for sid, r in references.items()}
        result[arm] = dict(
            mae=mean(mean(abs(errors[s]) for s in ids) for ids in groups.values()),
            rmse=math.sqrt(mean(mean(errors[s] ** 2 for s in ids) for ids in groups.values())))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True, type=Path)
    parser.add_argument('--recompute', action='store_true', help='Run saved models; otherwise replay predictions')
    parser.add_argument('--encoder', type=Path, help='Matching local Qwen encoder; required for recompute')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out', type=Path, help='New output directory, outside the source study')
    args = parser.parse_args(argv)
    if args.recompute and args.encoder is None:
        parser.error('--recompute requires --encoder')
    if args.out and (args.out.exists() or args.out.resolve().is_relative_to(args.study.resolve())):
        parser.error('--out must be a new directory outside the source study')
    m, identity, states, refs, check = load_study(args.study)
    predictions = (recompute(args.study, m, states, args.encoder, args.device)
                   if args.recompute else replay(args.study, states, check))
    metrics = summarize(states, refs, predictions)
    print('One frozen actor; shared prefixes and MC references; raw predictions.')
    print('Transformer: offline MC-return fit, NOT conventional PPO.')
    print(f'{len(states)} states; {len({s["question_id"] for s in states.values()})} questions; '
          f'{m["recipe"]["samples"]} MC samples/state')
    print(f'{"Estimator":<34} {"MAE":>9} {"RMSE":>9}')
    for arm, title in zip(ARMS, TITLES):
        print(f'{title:<34} {metrics[arm]["mae"]:>9.5f} {metrics[arm]["rmse"]:>9.5f}')
    rows = [dict(state_id=sid, question_id=s['question_id'],
                 mc_reference=refs[sid]['successes'] / refs[sid]['samples'],
                 **{arm: predictions[arm][sid] for arm in ARMS}) for sid, s in states.items()]
    print('\nFirst three states (MC reference, then estimator predictions):')
    for row in rows[:3]:
        print(row)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=False)
        write_json(args.out / 'summary.json', dict(identity, mode='recompute' if args.recompute else 'replay',
            scope='Offline fixed-policy comparison; MC-fitted transformer, not PPO.', metrics=metrics))
        with (args.out / 'predictions.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == '__main__':
    main()
