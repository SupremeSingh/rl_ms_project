"""Export offline probe evidence from saved predictions; no generation or fitting."""
import argparse
from collections import Counter
import csv
from html import escape
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import torch
from math_rl.critic_probe import metrics
from math_rl.large_probe import prefix_positions
from math_rl.provenance import sha256, write_json

ARMS = ('rms_hidden_concat', 'hidden_two_thirds', 'final')
FIELDS = ('y', 'question', 'position', 'response')


def read(path):
    return json.loads(path.read_text())


def verified_tensor(path, checksum):
    if sha256(path) != checksum:
        raise ValueError(f'Checksum mismatch: {path}')
    return torch.load(path, weights_only=True, map_location='cpu')


def metadata(source, arm, split):
    directory = source/'packed'/arm
    name = f'{split}-labels.pt'
    record = read(directory/'complete.json')
    data = verified_tensor(directory/name, record['files'][name])
    if any(len(data[f]) != record['counts'][split] for f in FIELDS):
        raise ValueError('Incomplete packed metadata')
    return {k: data[k].numpy() for k in FIELDS}


def summarize(y, predictions):
    """Mean metrics across independently fitted heads, never ensemble predictions."""
    predictions = np.atleast_2d(predictions)
    if predictions.shape[1] != len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError('Unaligned predictions or nonbinary labels')
    rows = [metrics(y, p) for p in predictions]
    keys = ('accuracy', 'brier', 'auroc', 'ece', 'outside_probability_range')
    return dict(prefixes=len(y), heads=len(rows),
                mean={k: None if any(r[k] is None for r in rows) else float(np.mean([r[k] for r in rows])) for k in keys},
                by_head=rows)


def paired_difference(y, a, b, question, repetitions=2000):
    """Average head losses first; resample paired questions, not prefixes."""
    groups = np.unique(question, return_inverse=True)[1]
    counts = np.bincount(groups)
    result = {}
    for metric in ('brier', 'accuracy'):
        if metric == 'brier':
            d = np.square(np.atleast_2d(a)-y).mean(0)-np.square(np.atleast_2d(b)-y).mean(0)
        else:
            d = ((np.atleast_2d(a)>=.5)==y).mean(0)-((np.atleast_2d(b)>=.5)==y).mean(0)
        per_question = np.bincount(groups, weights=d)/counts
        rng = np.random.default_rng(913)
        boot = [rng.choice(per_question, len(counts), replace=True).mean() for _ in range(repetitions)]
        result[metric] = dict(question_mean_difference=float(per_question.mean()),
            descriptive_95_interval=np.quantile(boot, [.025, .975]).tolist())
    return dict(questions=len(counts), **result)


def verify_test_trajectories(source, questions, data):
    """Recover realized lengths and check exact sampled-state ordering and labels."""
    expected = {k: [] for k in FIELDS}
    lengths, capped = [], []
    statuses = Counter(); attempted = retained = 0
    files = {}
    for q in questions:
        if q['split'] != 'test':
            continue
        path = source/'trajectories'/f'{q["index"]:05d}.json'
        rows = read(path)['responses']; files[path.name] = sha256(path)
        for response, row in enumerate(rows):
            attempted += 1; statuses[row['verifier_status']] += 1
            if row['verifier_status'] not in ('correct', 'incorrect') or not row['token_ids']:
                continue
            t = len(row['token_ids']); retained += 1
            pos = prefix_positions(t, q['index'], response)
            expected['y'].extend([float(row['score'])]*len(pos))
            expected['question'].extend([q['index']]*len(pos))
            expected['response'].extend([response]*len(pos))
            expected['position'].extend(pos)
            lengths.extend([t]*len(pos)); capped.extend([row['finish_reason']=='length']*len(pos))
    for field in FIELDS:
        if not np.array_equal(expected[field], data[field]):
            raise ValueError(f'Trajectory/prefix alignment differs: {field}')
    return np.array(lengths), np.array(capped), dict(attempted=attempted, retained=retained,
        excluded=attempted-retained, statuses=dict(statuses), trajectory_sha256=files)


def benchmark(source, sparse, screening, width, batch=256, repeats=10):
    """CPU cached-feature microbenchmark: excludes the LLM forward pass and I/O."""
    from math_rl.sparse_bottleneck import Bottleneck
    packed = source/'packed'/'rms_hidden_concat'/'test-x.npy'
    full = torch.from_numpy(np.array(np.load(packed, mmap_mode='r')[:batch], copy=True))
    single = torch.from_numpy(np.array(np.load(source/'packed/hidden_two_thirds/test-x.npy', mmap_mode='r')[:batch], copy=True))
    final = torch.from_numpy(np.array(np.load(source/'packed/final/test-x.npy', mmap_mode='r')[:batch], copy=True))
    operations = {'normalized_concat_readout': lambda: full@torch.ones(full.shape[1]),
                  'two_thirds_readout': lambda: single@torch.ones(single.shape[1]),
                  'final_readout': lambda: final@torch.ones(final.shape[1])}
    details = {}
    for kind in ('plain', 'sparse'):
        _, key, item = min((v['validation']['brier'], k, v) for k,v in screening['selected'].items() if k.startswith(kind+'-'))
        record = verified_tensor(sparse/item['path'], item['sha256'])
        model = Bottleneck(full.shape[1], width).eval().requires_grad_(False)
        model.load_state_dict(record['state'])
        w = torch.ones(width)
        def operation(model=model, r=record, w=w):
            return model.features((full-r['mean'])/r['scale'])@w
        operations[kind+'_compressor_and_readout'] = operation
        z = model.features((full-record['mean'])/record['scale'])
        details[kind] = dict(validation_selected_checkpoint=key, zero_fraction=float((z==0).float().mean()),
            projection_parameters=model.projection.weight.numel()+model.projection.bias.numel(),
            linear_readout_parameters=width+1, checkpoint_bytes=(sparse/item['path']).stat().st_size)
    timings = {}
    with torch.inference_mode():
        for name, operation in operations.items():
            for _ in range(3): operation()
            samples = []
            for _ in range(repeats):
                start=time.perf_counter(); operation(); samples.append(time.perf_counter()-start)
            timings[name] = dict(batch_size=len(full), median_batch_ms=float(np.median(samples)*1000),
                median_microseconds_per_prefix=float(np.median(samples)*1e6/len(full)))
    return dict(device='cpu', threads=torch.get_num_threads(), processor=platform.processor(),
        repeats=repeats, operations=timings, checkpoints=details,
        scope='Cached-feature CPU kernel timing only; raw controls are readout-only, bottlenecks include normalization and compression. '
              'Synthetic all-one readout weights. Excludes encoder, layer capture, transfers, I/O and fitting; not end-to-end latency.')


def svg_chart(path, results, metric, title):
    width,height=1100,100+len(results)*56
    s=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
       '<rect width="100%" height="100%" fill="white"/>',
       '<style>text{font:18px Arial,sans-serif;fill:#172b40}</style>',
       f'<text x="30" y="33" style="font-size:24px;font-weight:bold">{escape(title)}</text>']
    maxval=max(v['mean'][metric] for v in results.values())*1.12
    for i,(name,r) in enumerate(results.items()):
        y=64+i*56; value=r['mean'][metric]
        s += [f'<text x="30" y="{y+22}">{escape(name)}</text>',
              f'<rect x="430" y="{y}" width="{550*value/maxval:.2f}" height="30" fill="#32788b"/>',
              f'<text x="{440+550*value/maxval:.2f}" y="{y+22}">{value:.4f}</text>']
    s.append(f'<text x="30" y="{height-12}" style="font-size:14px">HEAD METRICS AVERAGED; SAME RETAINED TEST PREFIXES. DESCRIPTIVE, INSPECTED TEST SET.</text></svg>')
    path.write_text('\n'.join(s))


def export(source, sparse, out, run_benchmark=True):
    source,sparse,out = source.resolve(),sparse.resolve(),out.resolve()
    if out==source or source in out.parents or out==sparse or sparse in out.parents:
        raise ValueError('Output must be separate from both source studies')
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()): raise ValueError('Output must be empty; use a new directory')
    protocol=read(sparse/'protocol.json'); manifest=read(source/'manifest.json')
    if protocol['source_manifest'] != sha256(source/'manifest.json'):
        raise ValueError('Bottleneck/source manifest mismatch')
    if sha256(source/'questions.json') != manifest['questions_sha256']:
        raise ValueError('Source questions changed')
    predictions_path=sparse/'predictions.npz'
    with np.load(predictions_path, allow_pickle=False) as saved:
        predictions={k:saved[k].copy() for k in saved.files}
    for arm in ARMS:
        if sha256(source/'packed'/arm/'complete.json') != protocol['packed'][arm]:
            raise ValueError('Packed source manifest changed')
    screening=read(sparse/'screen.json')
    for item in screening['selected'].values():
        if sha256(sparse/item['path']) != item['sha256']:
            raise ValueError('Selected compressor changed')
    data=metadata(source, 'rms_hidden_concat', 'test')
    for arm in ARMS[1:]:
        control=metadata(source,arm,'test')
        if any(not np.array_equal(data[k],control[k]) for k in FIELDS):
            raise ValueError('Unmatched control test states')
    if not np.array_equal(predictions.pop('y'),data['y']) or not np.array_equal(predictions.pop('question'),data['question']):
        raise ValueError('Saved prediction labels are unaligned')
    questions=read(source/'questions.json')
    split_counts={}
    prior=None
    for split in ('train','val','test'):
        d=metadata(source,'rms_hidden_concat',split)
        if not np.isin(d['y'],[0,1]).all() or len(d['y'])%8: raise ValueError('Invalid labels/prefix count')
        qset={q['index'] for q in questions if q['split']==split}
        if not set(d['question']).issubset(qset): raise ValueError('Question split leakage')
        pairs=np.stack([d['question'],d['response']],axis=1)
        retained=len(np.unique(pairs,axis=0))
        if retained*8!=len(d['y']): raise ValueError('Expected eight prefixes per retained answer')
        split_counts[split]=dict(selected_questions=len(qset), retained_questions=len(np.unique(d['question'])),
            retained_answers=retained,prefixes=len(d['y']),retained_success_rate=float(d['y'].mean()))
        if split=='train':prior=float(d['y'].mean())
    lengths,capped,test_audit=verify_test_trajectories(source,questions,data)
    groups={}
    groups['constant_train_prior']=np.full((1,len(data['y'])),prior)
    groups['final/ridge']=predictions['final/ridge'][None]
    groups['two_thirds/ridge']=predictions['hidden_two_thirds/ridge'][None]
    groups['normalized_concat/ridge']=predictions['rms_hidden_concat/ridge'][None]
    for kind in ('plain','sparse'):
        groups[f'{kind}_{protocol["width"]}/ridge']=np.stack([predictions[f'{kind}-{seed}/ridge'] for seed in protocol['seeds']])
    # Same held-out states, but these neural heads were trained on sampled prefixes.
    neural=source/'fits/rms_hidden_concat/resnet10/selection.json'
    if neural.exists():
        selection=read(neural)['selected']
        groups['normalized_concat/resnet10']=np.stack([np.load(neural.parent/f'test-seed{seed}.npy',allow_pickle=False) for seed in selection])
    y=data['y']; results={k:summarize(y,p) for k,p in groups.items()}
    if neural.exists():
        expected=read(source/'summary.json')['results']['rms_hidden_concat/resnet10']['means']
        for field in ('accuracy','brier'):
            if not np.isclose(results['normalized_concat/resnet10']['mean'][field],expected[field],atol=1e-8,rtol=0):
                raise ValueError('Cached neural reference differs from source report')
    stored=read(sparse/'summary.json')['results']
    for key,p in predictions.items():
        measured=metrics(y,p)
        for field in ('accuracy','brier'):
            if not np.isclose(measured[field],stored[key][field],atol=1e-8,rtol=0):
                raise ValueError(f'Saved report/prediction mismatch: {key}/{field}')
    pos=data['position']; frac=pos/lengths
    masks={'question_only':pos==0,'tokens_1_32':(pos>0)&(pos<=32),
        'tokens_33_128':(pos>32)&(pos<=128),'tokens_129_512':(pos>128)&(pos<=512),'tokens_513_plus':pos>512,
        'fraction_first_third':(pos>0)&(frac<=1/3),'fraction_middle_third':(frac>1/3)&(frac<=2/3),
        'fraction_last_third':frac>2/3,'uncapped_answers':~capped,'capped_answers':capped}
    prefix={name:dict(questions=len(np.unique(data['question'][mask])),
        results={k:summarize(y[mask],p[:,mask]) for k,p in groups.items()}) for name,mask in masks.items() if mask.any()}
    sparse_key=f'sparse_{protocol["width"]}/ridge'; plain_key=f'plain_{protocol["width"]}/ridge'
    comparisons={f'{sparse_key}_minus_{b}':paired_difference(y,groups[sparse_key],groups[b],data['question'])
        for b in ('final/ridge','two_thirds/ridge','normalized_concat/ridge',plain_key)}
    # No threshold or representation is selected using this export.
    scope=('Offline eventual-verifier-correctness prediction on retained answers. Previously inspected test questions. '
        'Metrics averaged across feature-training seeds, not an ensemble; seeds share data. '
        'Ridge uses all-token fitting; neural reference uses sampled-prefix training. '
        'Intervals resample questions and are descriptive/unadjusted. Relative positions use realized future length for analysis only. '
        'Sparse and plain are nonlinear learned compressors followed by linear Ridge; no PPO claim or proven end-to-end speedup.')
    summary=dict(scope=scope,counts=split_counts,test_answer_audit=test_audit,train_prior=prior,
        results=results,by_prefix=prefix,paired_comparisons=comparisons,
        dimensions=dict(input=manifest['dimension']*len(manifest['layers']),bottleneck=protocol['width'],
            linear_parameters=protocol['width']+1,matrix_dimension=protocol['width']+1,
            projection_parameters=(manifest['dimension']*len(manifest['layers'])+1)*protocol['width'],
            float64_ridge_matrix_bytes={str(d):(d+1)**2*8 for d in
                (protocol['width'],manifest['dimension'],manifest['dimension']*len(manifest['layers']))}),
        source_report_counts=read(source/'summary.json').get('counts'),
        benchmark=benchmark(source,sparse,screening,protocol['width']) if run_benchmark else None,
        provenance=dict(source=str(source),sparse=str(sparse),files={str(p):sha256(p) for p in (
            source/'manifest.json', source/'questions.json', sparse/'protocol.json', sparse/'screen.json',
            sparse/'summary.json',predictions_path)},script_sha256=sha256(Path(__file__)),
            python=platform.python_version(),numpy=np.__version__,torch=torch.__version__))
    write_json(out/'summary.json',summary)
    rows=[]
    for group, items in [('overall',results)]+[(k,v['results']) for k,v in prefix.items()]:
        rows.extend(dict(group=group,representation=k,prefixes=v['prefixes'],heads=v['heads'],**v['mean']) for k,v in items.items())
    with (out/'metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    svg_chart(out/'accuracy.svg',results,'accuracy','EVENTUAL CORRECTNESS PREDICTION: ACCURACY')
    svg_chart(out/'brier.svg',results,'brier','EVENTUAL CORRECTNESS PREDICTION: BRIER (LOWER IS BETTER)')
    absolute={k:prefix[k]['results'][sparse_key] for k in
        ('question_only','tokens_1_32','tokens_33_128','tokens_129_512','tokens_513_plus') if k in prefix}
    svg_chart(out/'prefix-brier.svg',absolute,'brier','SPARSE RIDGE: PREDICTION ERROR BY PREFIX POSITION')
    lines=['# Offline probe evidence for a poster','',scope,'',
        '| Representation | Accuracy | Brier | AUROC | ECE |','|---|---:|---:|---:|---:|']
    for k,r in results.items():
        v=r['mean'];auc='NA' if v['auroc'] is None else f'{v["auroc"]:.4f}'
        lines.append(f'| {k} | {v["accuracy"]:.4f} | {v["brier"]:.5f} | {auc} | {v["ece"]:.4f} |')
    lines += ['', '## Data and interpretation','',f'Counts: `{json.dumps(split_counts)}`',
        f'Test answers: {test_audit["retained"]}/{test_audit["attempted"]} retained.',
        'Each retained answer contributes L=0 plus seven uniform nonterminal token-prefix draws with replacement.',
        'See metrics.csv for question-only, early-token, relative-position and capped/uncapped results.',
        'Calibration bins, per-head scores, paired intervals, exact counts and hashes are in summary.json.',
        'The bottleneck projection has millions of learned parameters; the final linear head has width + 1.',
        'Tenfold feature-width reduction is not tenfold inference speedup. CPU timing excludes the LLM.',
        'Raw Ridge outputs can lie outside [0,1]; raw Brier and ECE are retained; cross-entropy clips probabilities.',
        '', '## Before printing','',
        '- Review verifier labels and exclusions in the source audit.jsonl / responses.html.',
        '- Show early-prefix performance before claiming prediction before reasoning.',
        '- Label this as an inspected, verifier-conditional test set; head seeds are not new datasets.',
        '- Report full compressor costs alongside the small readout; avoid an end-to-end speed claim.',
        '- Comparisons across sampled-prefix neural training and all-token Ridge change the fitting distribution.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:len(results)+6]),flush=True)
    print(f'Poster evidence saved to {out}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path);p.add_argument('sparse',type=Path);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--no-benchmark',action='store_true');args=p.parse_args()
    torch.set_num_threads(8)
    export(args.source,args.sparse,args.out,not args.no_benchmark)


if __name__=='__main__': main()
