"""Bounded validation-selected sparse-reconstruction ablation on cached J-lens data."""
import argparse
import hashlib
import html
import json
import time
from pathlib import Path

import numpy as np
import torch

import compare_jspace_probes as original
from math_rl.critic_probe import fit_ridge, make_head, metrics, predict
from math_rl.jspace import pursuit_path, rotated_dictionary, vocabulary_directions
from math_rl.provenance import sha256, snapshot, write_json


def subset(data, question_limit, answer_limit):
    """Outcome-independent question selection; retain all prefixes of chosen answers."""
    questions = sorted(data['question'].unique().tolist(),
        key=lambda q: hashlib.sha256(f'jspace-refine-927:{q}'.encode()).hexdigest())
    if question_limit:
        questions = questions[:question_limit]
    keep = torch.zeros(len(data['y']), dtype=torch.bool)
    for q in questions:
        indices = data['question'] == q
        answers = sorted(data['response'][indices].unique().tolist())[:answer_limit]
        for answer in answers:
            keep |= indices & (data['response'] == answer)
    if not keep.any():
        raise ValueError('Empty selected split')
    return {k: v[keep] for k, v in data.items()}


def paired(left, right, y, questions):
    delta = (left-y)**2 - (right-y)**2
    values = np.array([delta[questions == q].mean() for q in np.unique(questions)])
    rng = np.random.default_rng(912)
    interval = np.quantile([rng.choice(values, len(values), replace=True).mean()
                           for _ in range(2000)], [.025, .975]).tolist()
    return dict(mean=float(values.mean()), interval_95=interval, questions=len(values))


def fit_head(folder, train, val):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/'ridge.pt'
    if path.exists():
        return torch.load(path, weights_only=True)
    mean = train['x'].mean(0)
    scale = train['x'].std(0, unbiased=False).clamp_min(1e-5)
    fitted = fit_ridge(train, val, mean, scale)
    record = dict(**fitted, mean=mean, scale=scale)
    original.frozen.save_tensor(path, record)
    return record


def prediction(record, data):
    head = make_head('linear_value', data['x'].shape[1])
    head.load_state_dict(record['state'])
    return predict(head, 'linear_value', data, record['mean'], record['scale'], torch.device('cpu'))


def features(data, final, component, arm):
    if arm == 'hidden': x = data['x']
    elif arm == 'final': x = final['x']
    elif arm == 'hidden_final': x = torch.cat((data['x'], final['x']), 1)
    elif arm in ('jspace', 'random'): x = component
    elif arm in ('jspace_final', 'random_final'): x = torch.cat((component, final['x']), 1)
    else: raise ValueError(arm)
    return dict(data, x=x)


def reconstruct_split(out, split, data, dictionary, methods, budgets, batch_size, tokenizer):
    outputs = {}
    for method in methods:
        for control in ('jspace', 'random'):
            names = [f'{method}-{k}-{control}' for k in budgets]
            paths = [out/'features'/f'{split}-{name}.pt' for name in names]
            if all(p.exists() for p in paths) and all(
                    (out/'diagnostics'/f'{split}-{name}.json').exists() for name in names):
                for name, path in zip(names, paths):
                    diagnostic = json.loads((out/'diagnostics'/f'{split}-{name}.json').read_text())
                    if diagnostic.get('features_sha256') != sha256(path):
                        raise ValueError(f'Cached reconstruction changed: {path}')
                    outputs[name] = torch.load(path, weights_only=True)['x']
                continue
            selected_dictionary = dictionary if control == 'jspace' else rotated_dictionary(dictionary)
            chunks = {k: [] for k in budgets}
            diagnostics = {k: [] for k in budgets}
            readouts = []
            started = time.perf_counter()
            for start in range(0, len(data['y']), batch_size):
                h = data['x'][start:start+batch_size].to(dictionary.device)
                results = pursuit_path(h, selected_dictionary, budgets, method)
                for k, (value, ids, weights) in results.items():
                    chunks[k].append(value.cpu())
                    residual = h-value
                    fractions = residual.square().sum(1)/h.square().sum(1).clamp_min(1e-12)
                    for row in range(len(h)):
                        positive = weights[row] > 1e-8
                        active = ids[row][positive].unique().numel()
                        diagnostics[k].append((float(fractions[row]), active))
                    if start == 0:
                        for row in range(min(8, len(h))):
                            idx = start+row
                            readouts.append(dict(budget=k, question=int(data['question'][idx]),
                                response=int(data['response'][idx]), prefix=int(data['position'][idx]),
                                token_ids=ids[row].cpu().tolist(),
                                tokens=[tokenizer.decode([int(t)]) for t in ids[row]],
                                coefficients=weights[row].cpu().tolist(),
                                residual_energy_fraction=float(fractions[row])))
                print(f'{split}/{method}/{control}: {min(start+batch_size,len(data["y"]))}/{len(data["y"])}', flush=True)
            elapsed = time.perf_counter()-started
            for k, name, path in zip(budgets, names, paths):
                values = np.asarray(diagnostics[k])
                # This is uncentered reconstruction error, not conventional explained variance.
                x = torch.cat(chunks[k])
                original.frozen.save_tensor(path, dict(x=x))
                diagnostic = dict(
                    residual_energy_fraction_mean=float(values[:, 0].mean()),
                    residual_energy_fraction_quantiles=np.quantile(values[:, 0], [.1,.5,.9]).tolist(),
                    active_distinct_atoms_mean=float(values[:, 1].mean()),
                    active_distinct_atoms_quantiles=np.quantile(values[:, 1], [.1,.5,.9]).tolist(),
                    shared_path_seconds=elapsed, shared_budgets=budgets,
                    readouts=[r for r in readouts if r['budget']==k], features_sha256=sha256(path))
                original.frozen.atomic_json(out/'diagnostics'/f'{split}-{name}.json', diagnostic)
                outputs[name] = x
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--train-questions', type=int, default=400)
    parser.add_argument('--val-questions', type=int, default=100)
    parser.add_argument('--answers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    if min(args.train_questions, args.val_questions, args.answers, args.batch_size) < 1:
        parser.error('Budgets must be positive')
    source, out = args.source.resolve(), args.out.resolve()
    if source == out: parser.error('Use a new output directory')
    environment = original.installed_lens()
    manifest = json.loads((source/'manifest.json').read_text())
    if environment != manifest['environment']:
        raise ValueError('Use the same pinned environment as the original run')
    extraction = json.loads((source/'extraction.json').read_text())
    info = json.loads((source/'lens-metadata.json').read_text())
    inputs = {str(source/p): digest for p, digest in extraction['files'].items()
              if p.startswith(('two_thirds/', 'final/'))}
    for representation in ('two_thirds','final'):
        for split in ('train','val','test'):
            if str(source/representation/f'{split}.pt') not in inputs:
                raise ValueError('Missing cached feature provenance')
    inputs[str(source/'lens.pt')] = info['sha256']
    inputs[str(source/'manifest.json')] = sha256(source/'manifest.json')
    for representation in ('final','two_thirds','jspace','jspace_final'):
        path=source/representation/'summary.json'
        inputs[str(path)]=sha256(path)
    for name, digest in manifest['model_hashes'].items():
        inputs[str(original.MODEL/name)] = digest
    for path, digest in inputs.items():
        if sha256(path) != digest: raise ValueError(f'Changed source artifact: {path}')
    config = dict(source=str(source), smoke=args.smoke,
        train_questions=4 if args.smoke else args.train_questions,
        val_questions=3 if args.smoke else args.val_questions,
        test_questions=3 if args.smoke else 0, answers=2 if args.smoke else args.answers,
        budgets=[2, 3] if args.smoke else [16, 25, 64], methods=['matching','gradient'],
        batch_size=args.batch_size, gradient_refinements=4)
    locked = dict(config=config, inputs=inputs, environment=environment, code=snapshot(original.ROOT)['files'])
    if out.exists():
        if json.loads((out/'manifest.json').read_text()) != locked:
            raise ValueError('Resume provenance changed; use a fresh output directory')
    else:
        out.mkdir(parents=True)
        write_json(out/'manifest.json', locked)
    for name in ('features', 'heads', 'diagnostics'):
        (out/name).mkdir(exist_ok=True)
    started = time.perf_counter()
    state = dict(state='running', stage='validation')
    write_json(out/'status.json', state)
    try:
        # Reuse the official fitted Jacobian and cached states. No model forward passes.
        import jlens
        from safetensors import safe_open
        from transformers import AutoTokenizer
        if not torch.cuda.is_available(): raise RuntimeError('Run on a GPU')
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        import os
        if os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE') == '1':
            raise RuntimeError('Unset TF32 override before starting Python')
        tensors = {}
        for path in original.MODEL.glob('*.safetensors'):
            with safe_open(path, framework='pt', device='cpu') as handle:
                for name in ('lm_head.weight', 'model.norm.weight'):
                    if name in handle.keys(): tensors[name] = handle.get_tensor(name)
        if len(tensors) != 2: raise ValueError('Expected explicit Qwen unembedding and final norm weights')
        lens = jlens.JacobianLens.load(str(source/'lens.pt'))
        dictionary = vocabulary_directions(tensors['lm_head.weight'].cuda(),
            tensors['model.norm.weight'].cuda(), lens.jacobians[info['source_layer_zero_based']].cuda())
        del tensors, lens
        tokenizer = AutoTokenizer.from_pretrained(original.MODEL, local_files_only=True)
        datasets, finals, transformed = {}, {}, {}

        def load_split(split):
            data = torch.load(source/'two_thirds'/f'{split}.pt', weights_only=True)
            final = torch.load(source/'final'/f'{split}.pt', weights_only=True)
            for key in ('question','response','position','y'):
                if not torch.equal(data[key],final[key]): raise ValueError('Unpaired source features')
            limit = config[f'{split}_questions']
            return subset(data,limit,config['answers']), subset(final,limit,config['answers'])

        for split in ('train','val'):
            datasets[split], finals[split] = load_split(split)
            transformed[split] = reconstruct_split(out, split, datasets[split], dictionary,
                config['methods'], config['budgets'], args.batch_size, tokenizer)
        candidates = []
        records = {}
        for arm in ('hidden','final','hidden_final'):
            records[arm] = fit_head(out/'heads'/arm, *[
                features(datasets[s], finals[s], None, arm) for s in ('train','val')])
        for method in config['methods']:
            for k in config['budgets']:
                for arm in ('jspace','jspace_final','random','random_final'):
                    name = f'{method}-{k}-{arm}'
                    component = 'random' if arm.startswith('random') else 'jspace'
                    records[name] = fit_head(out/'heads'/name, *[
                        features(datasets[s], finals[s], transformed[s][f'{method}-{k}-{component}'], arm)
                        for s in ('train','val')])
                record = records[f'{method}-{k}-jspace']
                candidates.append(dict(method=method, budget=k, validation_brier=record['validation_brier']))
        selected = min(candidates, key=lambda c: c['validation_brier'])
        # Selection is written before loading any test labels/features; one primary J-only criterion.
        selection = dict(selected=selected, candidates=candidates,
            scope='Ridge J-only validation Brier selects method/budget; same setting used for concatenation and random controls.')
        if (out/'selection.json').exists() and json.loads((out/'selection.json').read_text()) != selection:
            raise ValueError('Locked validation selection changed')
        write_json(out/'selection.json', selection)
        state['stage'] = 'locked-test'; write_json(out/'status.json', state)
        del transformed
        test, final = load_split('test')
        method, k = selected['method'], selected['budget']
        transformed = reconstruct_split(out, 'test', test, dictionary, [method], [k], args.batch_size, tokenizer)
        results, predictions = {}, {}
        for arm in ('hidden','final','hidden_final','jspace','jspace_final','random','random_final'):
            name = arm if arm in ('hidden','final','hidden_final') else f'{method}-{k}-{arm}'
            component = None if name==arm else transformed[f'{method}-{k}-'+('random' if arm.startswith('random') else 'jspace')]
            predictions[arm] = prediction(records[name], features(test, final, component, arm))
            np.save(out/f'{arm}-predictions.npy', predictions[arm])
            results[arm] = metrics(test['y'].numpy(), predictions[arm])
        pairs = {f'{a}_minus_{b}': paired(predictions[a], predictions[b], test['y'].numpy(), test['question'].numpy())
                 for a,b in [('jspace','hidden'),('jspace_final','final'),('jspace_final','hidden_final'),
                             ('jspace','random'),('jspace_final','random_final')]}
        counts = {s:dict(prefixes=len(d['y']), questions=len(d['question'].unique()))
                  for s,d in dict(datasets,test=test).items()}
        # Inspect old convergence without selecting new candidates on old test scores.
        fitting_audit = {}
        for representation in ('final','two_thirds','jspace','jspace_final'):
            old = json.loads((source/representation/'summary.json').read_text())
            fitting_audit[representation] = [dict(kind=r['kind'],seed=r['seed'],
                best_epoch=r['best_epoch'],epochs_run=r['epochs_run'],
                best_at_budget_limit=r['best_at_budget_limit']) for r in old['heads']]
        write_json(out/'original-fitting-audit.json', fitting_audit)
        scope = ('Bounded ridge-only reconstruction study; reused inspected questions and verifier labels. '
                 'Our projected nonnegative gradient-pursuit implementation, not official decomposition code. '
                 'No fresh benchmark, causal claim, LSTD or PPO update. Intervals exploratory/unadjusted. '
                 'Selection uses validation only; test evaluates one locked method/budget.')
        write_json(out/'summary.json', dict(selection=selection,results=results,paired=pairs,
            counts=counts,scope=scope,smoke=args.smoke,head_fits={name:{key:value for key,value in r.items()
                if key not in ('state','mean','scale')} for name,r in records.items()}))
        lines = [scope, f'Selected: {method}, budget {k}', f'Counts: {counts}',
                 'representation       Brier      accuracy']
        lines += [f'{name:20} {r["brier"]:.5f}    {r["accuracy"]:.3f}' for name,r in results.items()]
        lines += [f'{name}: {value}' for name,value in pairs.items()]
        (out/'report.txt').write_text('\n'.join(lines)+'\n')
        review = ['<!doctype html><meta charset="utf-8"><title>J-space pursuit diagnostics</title>',
                  '<h1>Reconstruction readouts</h1><p>First eight prefixes per split/configuration; '
                  'not correctness explanations. Random dictionary token labels have no semantic meaning.</p>']
        qmap = {q['original_index']:q for q in manifest['questions']}
        for path in sorted((out/'diagnostics').glob('*.json')):
            diagnostic = json.loads(path.read_text())
            review.append('<h2>'+html.escape(path.stem)+'</h2>')
            for readout in diagnostic['readouts']:
                question=qmap[readout['question']]
                trajectory=Path(manifest['source'])/'trajectories'/f'{readout["question"]:05d}.json'
                if sha256(trajectory) != manifest['source_hashes'][str(trajectory)]:
                    raise ValueError('Source trajectory changed during response review')
                row=json.loads(trajectory.read_text())['responses'][readout['response']]
                review.append('<details><summary>'+html.escape(question['id']+f' / prefix {readout["prefix"]}')+
                    '</summary><pre style="white-space:pre-wrap">'+html.escape(question['question']+'\n'+row['response']+
                    '\n'+json.dumps(readout,indent=2,ensure_ascii=False))+'</pre></details>')
        (out/'responses.html').write_text('\n'.join(review))
        state['state']='complete'
    except BaseException as exc:
        state.update(state='failed',error=str(exc)); raise
    finally:
        state['invocation_seconds']=time.perf_counter()-started
        write_json(out/'status.json',state)


if __name__ == '__main__':
    main()
