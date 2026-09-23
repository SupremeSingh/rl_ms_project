"""Fit three GSM8K LSTD traces and compare on two existing, matched rollout caches."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
from analyze_lstd import diagnostics, metadata, paired, report_text
from fit_lstd import read_shard, run as fit
from frozen_critics import atomic_json
from math_rl.critic_probe import PREFIX_SAMPLING, assessment, sample_shard
from math_rl.lstd import predict
from math_rl.provenance import sha256

LAMBDAS = [.85, .9, .95]
ALPHAS = [0., 1e-6, 1e-5, 1e-4, .001, .01, .1, 1.]
METHODS = ['lstd-0.85', 'lstd-0.9', 'lstd-0.95', 'ridge']


def load_cache(path, source, protocol):
    manifest = json.loads((path / 'manifest.json').read_text())
    report = json.loads((path / 'summary.json').read_text())
    if manifest['protocol'] != protocol or Path(manifest['source']).resolve() != source:
        raise ValueError('Wrong evaluation cache or training source')
    if manifest['config']['prefix_sampling'] != PREFIX_SAMPLING:
        raise ValueError('Incompatible prefix sampling')
    original_fit = Path(manifest['fit'])
    if sha256(original_fit / 'manifest.json') != manifest['files'][str(original_fit / 'manifest.json')]:
        raise ValueError('Original fitting manifest changed')
    original = json.loads((original_fit / 'manifest.json').read_text())
    for name, expected in original['source_hashes'].items():
        if sha256(source / name) != expected:
            raise ValueError(f'Original training source changed: {name}')
    questions = manifest['questions']
    training = json.loads((source / 'questions.json').read_text())['questions']
    forbidden = {q['id'] for q in training if q['split'] != 'test'}
    if any(q['split'] != 'test' or q['id'] in forbidden for q in questions):
        raise ValueError('Evaluation questions overlap training/validation')
    if len({q['id'] for q in questions}) != len(questions):
        raise ValueError('Duplicate evaluation question')
    data = torch.load(path / 'test.pt', weights_only=True, map_location='cpu')
    offset, hashes = 0, {}
    for i, q in enumerate(questions):
        shard, _, fingerprint = read_shard(path, i, q)
        if fingerprint != report['feature_hashes'][str(i)]:
            raise ValueError('Evaluation features changed since original report')
        hashes[str(i)] = fingerprint
        if not len(shard['rewards']):
            continue
        probe = sample_shard(shard, q['source_index'])
        count = len(probe['y'])
        if any(not torch.equal(value, data[key][offset:offset + count]) for key, value in probe.items()):
            raise ValueError('Saved test probes differ from original features')
        offset += count
    if offset != len(data['y']) or not offset:
        raise ValueError('Incomplete test cache')
    meta, provenance = metadata(path, questions, data)
    files = {str(path / name): sha256(path / name) for name in ('manifest.json', 'summary.json', 'test.pt')}
    return dict(path=path, questions=questions, data=data, meta=meta,
                provenance=provenance, hashes=hashes, files=files)


def evaluate(cache, fitted, out, baseline):
    data = cache['data']
    results, predictions = {}, {}
    reference = torch.load(fitted / 'ridge.pt', weights_only=True, map_location='cpu')
    out.mkdir(exist_ok=True)
    for method in METHODS:
        head = torch.load(fitted / f'{method}.pt', weights_only=True, map_location='cpu')
        if not all(torch.equal(head[key], reference[key]) for key in ('mean', 'scale')):
            raise ValueError('Methods have different feature normalization')
        predictions[method] = predict(head['weights'], data, head['mean'], head['scale'])
        np.save(out / f'{method}-predictions.npy', predictions[method])
        results[method] = dict(alpha=head['alpha'], trace_lambda=head['trace_lambda'],
            validation_brier=head['validation_brier'], test=assessment(data, predictions[method], baseline))
    for method in METHODS[:-1]:
        diagnostic = diagnostics(cache['path'], cache['questions'], data,
                                 predictions[method], predictions['ridge'])
        atomic_json(out / f'{method}-length.json', diagnostic)
        (out / f'{method}-length.txt').write_text(report_text(diagnostic))
        results[method]['versus_ridge'] = diagnostic['groups']['overall']
        if all('level' in q for q in cache['questions']):
            results[method]['levels'] = {}
            for level in (4, 5):
                ids = [q['source_index'] for q in cache['questions'] if q['level'] == level]
                results[method]['levels'][str(level)] = paired(cache['meta'], predictions[method],
                    predictions['ridge'], np.isin(cache['meta']['question'], ids))
    return dict(methods=results, answers=cache['provenance'])


def run(source, gsm8k, math, out):
    source, gsm8k, math, out = [p.resolve() for p in (source, gsm8k, math, out)]
    if any(out == p or p in out.parents or out in p.parents for p in (source, gsm8k, math)):
        raise ValueError('Choose a separate output directory')
    # Verify caches before doing the expensive fit. Original artifacts are read-only.
    caches = dict(gsm8k_new_answers=load_cache(gsm8k, source, 'lstd-seed-validation-v1'),
                  math_transfer=load_cache(math, source, 'lstd-math-hard-v1'))
    tracked = [Path(__file__).resolve(), ROOT / 'scripts/fit_lstd.py', ROOT / 'scripts/analyze_lstd.py',
               ROOT / 'src/math_rl/lstd.py', ROOT / 'src/math_rl/critic_probe.py']
    manifest = dict(protocol='lstd-trace-comparison-v1', source=str(source), lambdas=LAMBDAS, alphas=ALPHAS,
        code={str(p): sha256(p) for p in tracked}, versions=dict(torch=str(torch.__version__), numpy=np.__version__),
        inputs={name: dict(files=c['files'], features=c['hashes']) for name, c in caches.items()},
        scope='Exploratory reuse of inspected test sets. New GSM8K answers on the same questions; MATH transfer. '
              'Alpha selected on original GSM8K validation only. Report every requested lambda; no test selection.')
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        if json.loads((out / 'manifest.json').read_text()) != manifest:
            raise ValueError('Resume requires unchanged data, code and settings')
    elif any(out.iterdir()):
        raise ValueError('Output must be empty or a compatible comparison run')
    atomic_json(out / 'manifest.json', manifest)
    started = time.monotonic()
    atomic_json(out / 'status.json', dict(stage='fit', state='running'))
    try:
        fit(source, out / 'fit', ALPHAS, LAMBDAS)
        atomic_json(out / 'status.json', dict(stage='evaluate', state='running'))
        baseline = torch.load(source / 'normalization.pt', weights_only=True)['baseline']
        results = {name: evaluate(cache, out / 'fit', out / name, baseline) for name, cache in caches.items()}
        atomic_json(out / 'summary.json', dict(scope=manifest['scope'], datasets=results,
            uncertainty='Paired question intervals are descriptive, not adjusted for multiple comparisons.',
            limitations='Conditional on retained answers; original caps and verifier exclusions preserved. '
                        'No new trajectories, MATH fitting, or PPO updates.'))
        lines = [manifest['scope'], 'Brier lower is better; accuracy predicts answer correctness.']
        for dataset, result in results.items():
            lines += ['', dataset, 'method      alpha      Brier    accuracy   paired difference vs ridge [95% interval]']
            for method, row in result['methods'].items():
                paired_result = row.get('versus_ridge')
                difference = (f"{paired_result['question_mean_difference']:+.6f} "
                    f"{paired_result['question_bootstrap_95pct_interval']}" if paired_result else 'reference')
                lines.append(f"{method:11} {row['alpha']:<10g} {row['test']['brier']:.5f}  "
                             f"{row['test']['accuracy']:.3f}      {difference}")
        lines += ['Per-level comparisons in summary.json; length diagnostics in dataset subdirectories.',
                  'Exploratory intervals; no multiplicity adjustment. Conditional on verifiable answers. No PPO claim.']
        (out / 'report.txt').write_text('\n'.join(lines) + '\n')
    except Exception as exc:
        atomic_json(out / 'status.json', dict(state='failed', error=str(exc)))
        raise
    atomic_json(out / 'status.json', dict(state='complete', seconds=time.monotonic() - started))
    print((out / 'report.txt').read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('gsm8k', type=Path)
    parser.add_argument('math', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.gsm8k, args.math, args.out)


if __name__ == '__main__':
    main()
