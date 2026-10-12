"""Visualize cached J-space/hidden critics and direct official J-lens readouts.

No generation, lens fitting, critic fitting, or policy updates. Validation only by
default. Selected question order is outcome-independent; display filters are not
used for the aggregate diagnostics.
"""
import argparse
import json
import os
from pathlib import Path
import time

import torch

import compare_jspace_probes as original
from math_rl.jspace import load_readout_weights
from math_rl.jspace_inspection import (CONCEPTS, association, concept_ids, direct_readout,
                                      html_payload, linear_contributions, visible_mentions)
from math_rl.provenance import sha256, snapshot, write_json


def run(args):
    import jlens
    from transformers import AutoTokenizer

    source = args.source.resolve()
    manifest = json.loads((source/'manifest.json').read_text())
    if json.loads((source/'status.json').read_text())['state'] != 'complete':
        raise ValueError('Use a completed geometric study')
    if original.installed_lens() != manifest['environment']:
        raise ValueError('Use the original pinned J-lens environment')
    if os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE') == '1':
        raise ValueError('Unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    out = args.out.resolve()
    if out.exists():
        raise ValueError('Use a new output directory; source and previous views stay unchanged')
    out.mkdir(parents=True)
    inputs = {}
    def checked(path, digest=None):
        path = Path(path)
        actual = sha256(path)
        if digest is not None and actual != digest:
            raise ValueError(f'Changed source: {path}')
        inputs[str(path)] = actual
        return path
    checked(source/'manifest.json')
    checked(source/'extraction.json')
    for path, digest in manifest['inputs'].items():
        # Check all model/tokenizer inputs; trajectory files below are checked when used.
        if '/models/' in path:
            checked(path, digest)
    old = Path(manifest['source'])
    checked(old/'manifest.json', manifest['inputs'][str(old/'manifest.json')])
    blocks = [layer+1 for layer in manifest['layers']]
    baseline = (2*(manifest['target_layer']+1)+2)//3
    blocks = sorted(set(blocks+[baseline]))
    lenses = {}
    for folder in (source, old):
        meta = json.loads(checked(folder/'lens-metadata.json').read_text())
        lenses[folder] = jlens.JacobianLens.load(str(checked(folder/'lens.pt', meta['sha256'])))
    tokenizer = AutoTokenizer.from_pretrained(original.MODEL, local_files_only=True)
    model_config = json.loads(checked(original.MODEL/'config.json').read_text())
    weights = load_readout_weights(original.MODEL)
    unembedding = weights['lm_head.weight'].to(device)
    gain = weights['model.norm.weight'].to(device)
    del weights
    groups, coverage = concept_ids(tokenizer, len(unembedding))
    if any(not ids for ids in groups.values()):
        raise ValueError('Concept group has no exact single-token coverage')
    heads = {name: torch.load(checked(source/name/'heads/ridge.pt'), weights_only=True)
             for name in ('hidden_two_thirds', 'jspace_concat')}
    questions = [(i,q) for i,q in enumerate(manifest['questions']) if q['split'] == args.split]
    if args.questions:
        questions = questions[:args.questions]
    if not questions:
        raise ValueError('No questions selected')
    answers = []; diagnostics = []; attempted = retained = 0
    started = time.perf_counter()
    for i,q in questions:
        marker = json.loads(checked(source/f'question-{i:05d}.json').read_text())
        if marker['id'] != q['id'] or marker['split'] != args.split:
            raise ValueError('Question metadata mismatch')
        trajectory_path = Path(manifest['trajectory_source'])/'trajectories'/f'{q["original_index"]:05d}.json'
        trajectory = json.loads(checked(trajectory_path, manifest['inputs'][str(trajectory_path)]).read_text())
        if trajectory['question_id'] != q['id']:
            raise ValueError('Trajectory question mismatch')
        attempted += marker['attempted']; retained += marker['retained']
        def shard(name):
            relative = f'{name}/sampled/{i:05d}.pt'
            data = torch.load(checked(source/relative, marker['files'][relative]), weights_only=True)
            return data
        baseline_data = shard('hidden_two_thirds')
        if not len(baseline_data['y']):
            continue
        data = {block: baseline_data if block == baseline else shard(f'hidden-block{block}') for block in blocks}
        jdata = shard('jspace_concat')
        for item in [*data.values(), jdata]:
            for key in ('y','position','response','question'):
                if not torch.equal(item[key], baseline_data[key]):
                    raise ValueError('Unpaired feature rows')
        predictions, contributions = linear_contributions(heads['hidden_two_thirds'], baseline_data['x'])
        jpredictions, _ = linear_contributions(heads['jspace_concat'], jdata['x'], count=1)
        readouts = {}
        for block, item in data.items():
            layer = block-1
            lens = lenses[source] if layer in lenses[source].jacobians else lenses[old]
            readouts[block] = []
            for h in item['x'].split(args.batch_size):
                h = h.to(device).float()
                transported = h if layer == manifest['target_layer'] else lens.transport(h, layer)
                ids, probabilities, masses = direct_readout(transported, None, unembedding, gain,
                    model_config.get('rms_norm_eps', 1e-6), groups)
                for row in range(len(h)):
                    readouts[block].append(dict(
                        top=[dict(token=tokenizer.decode([t]), token_id=t, probability=p)
                             for t,p in zip(ids[row].tolist(), probabilities[row].tolist())],
                        mass={name: float(value[row]) for name,value in masses.items()}))
        for ri in sorted(baseline_data['response'].unique().tolist()):
            row = trajectory['responses'][ri]
            indices = torch.where(baseline_data['response'] == ri)[0].tolist()
            positions = {}
            for index in indices:
                p = int(baseline_data['position'][index])
                if p in positions:  # Prefix sampling originally uses replacement.
                    continue
                prefix = tokenizer.decode(row['token_ids'][:p], skip_special_tokens=False)
                mentions = visible_mentions(q['question']+'\n'+prefix)
                layer_readouts = {}
                for block in blocks:
                    r = dict(readouts[block][index])
                    sparse = next((r for r in marker['readouts'] if r['block']==block and r['response_index']==ri), None)
                    if sparse is not None:
                        k = sparse['prefix_lengths'].index(p)
                        r['sparse'] = sorted([dict(token=t, coefficient=w) for t,w in
                            zip(sparse['tokens'][k], sparse['coefficients'][k]) if w>0],
                            key=lambda a: -a['coefficient'])[:12]
                        r['residual_energy'] = sparse['residual_energy_fraction'][k]
                    layer_readouts[str(block)] = r
                    diagnostics.append(dict(question=q['id'],response=ri,position=p,block=block,
                        correct=int(row['score']), mass=r['mass'], mentions=mentions))
                positions[p] = dict(position=p, prefix=prefix, hidden_value=float(predictions[index]),
                    jspace_value=float(jpredictions[index]), contributions=contributions[index],
                    layers=layer_readouts, mentions=mentions)
            answers.append(dict(id=q['id'], response=ri, question=q['question'],
                correct=int(row['score']), status=row['verifier_status'], full_response=row['response'],
                reference=q['ground_truth'], points=[positions[p] for p in sorted(positions)]))
        print(f'Inspected {len(answers)} retained answers; question {q["id"]}', flush=True)
    scope = ('Exploratory read-only diagnostics on saved verifier-retained answers. Default validation split; '
        'these questions already selected critic regularization. Direct lens mass is not a correctness probability. '
        'Eight cached prefix draws per answer, deduplicated for display/association; no dense error-onset localization. '
        'Word groups fixed before inspection; no threshold tuning, new critic fitting or causal claim. '
        'No-visible-mention filtering is diagnostic only. Intervals unadjusted and clustered by question.')
    summary = dict(scope=scope, split=args.split, questions=len(questions), attempted=attempted,
        retained=retained, seconds=time.perf_counter()-started, blocks=blocks, baseline_block=baseline,
        concepts=CONCEPTS, token_coverage=coverage, associations=association(diagnostics))
    write_json(out/'summary.json', summary)
    write_json(out/'inspection.json', dict(summary=summary,answers=answers))
    write_json(out/'manifest.json', dict(inputs=inputs, code=snapshot(original.ROOT)['files'],
        environment=original.installed_lens(), config=vars(args)|{'source':str(source),'out':str(out)}))
    template = (original.ROOT/'scripts/templates/jspace_inspector.html').read_text()
    (out/'inspector.html').write_text(template.replace('__PAYLOAD__', html_payload(dict(summary=summary,answers=answers))))
    lines = [scope, f'{args.split}: {len(questions)} questions, {retained}/{attempted} retained answers',
             'Failure AUROC > 0.5 means larger concept mass ranks wrong answers higher; not calibrated.']
    lines += [f'{key}: {value}' for key,value in summary['associations'].items()]
    (out/'report.txt').write_text('\n'.join(lines)+'\n')
    write_json(out/'status.json', dict(state='complete'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='Completed geometric J-space output')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--split', choices=('train','val','test'), default='val')
    parser.add_argument('--questions', type=int, default=0, help='0 means all questions in the split')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    args = parser.parse_args()
    if args.questions<0 or args.batch_size<1:
        parser.error('Invalid budget')
    if args.out.exists():
        parser.error('Use a new output directory')
    try:
        run(args)
    except Exception as exc:
        # Never write into an existing source or prior successful output on failure.
        if args.out.exists() and args.out.resolve()!=args.source.resolve():
            write_json(args.out/'status.json', dict(state='failed',error=str(exc)))
        raise


if __name__ == '__main__':
    main()
