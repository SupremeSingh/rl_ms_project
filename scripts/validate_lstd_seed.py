"""New-generation-seed evaluation of LOCKED critics on the original test questions."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from importlib.metadata import version

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
from analyze_lstd import diagnostics, report_text
from fit_lstd import read_shard
from frozen_critics import atomic_json, extract, generate, save_tensor
from math_rl.critic_probe import PREFIX_SAMPLING, assessment, sample_shard
from math_rl.lstd import predict
from math_rl.ppo_reward import verifier_command
from math_rl.provenance import sha256


def evaluate(out, manifest, questions):
    fit = Path(manifest['fit'])
    source = Path(manifest['source'])
    parts, hashes = [], {}
    for i, q in enumerate(questions):
        shard, _, fingerprint = read_shard(out, i, q)
        hashes[str(i)] = fingerprint
        if len(shard['rewards']):
            parts.append(sample_shard(shard, q['source_index']))
    if not parts:
        raise ValueError('No usable fresh answers; inspect verifier statuses')
    data = {key: torch.cat([p[key] for p in parts]) for key in parts[0]}
    save_tensor(out / 'test.pt', data)
    norm = torch.load(source / 'normalization.pt', weights_only=True, map_location='cpu')
    results, predictions = {}, {}
    for name, file in (('lstd', 'selected-lstd.pt'), ('ridge', 'ridge.pt')):
        if sha256(fit / file) != manifest['files'][str(fit / file)]:
            raise ValueError('Locked critic changed')
        head = torch.load(fit / file, weights_only=True, map_location='cpu')
        if not torch.equal(head['mean'].double(), norm['mean'].double()) or not torch.equal(head['scale'].double(), norm['scale'].double()):
            raise ValueError('Critic normalization differs from source')
        predictions[name] = predict(head['weights'], data, head['mean'], head['scale'])
        np.save(out / f'{name}-predictions.npy', predictions[name])
        results[name] = dict(alpha=head['alpha'], trace_lambda=head.get('trace_lambda'),
                            test=assessment(data, predictions[name], norm['baseline']))
    diagnostic = diagnostics(out, questions, data, predictions['lstd'], predictions['ridge'])
    atomic_json(out / 'length-summary.json', diagnostic)
    (out / 'length-report.txt').write_text(report_text(diagnostic))
    old = json.loads((fit / 'summary.json').read_text())
    report = dict(scope='New answers on the SAME test questions; locked critics, no refitting or retuning.',
        seed=manifest['config']['generation_seed'], questions=len(questions), selected_lstd=manifest['selected_lstd'],
        results=results, paired=diagnostic['groups']['overall'],
        original_paired_difference=old['paired_question_brier_selected_lstd_minus_ridge'],
        answers=diagnostic['provenance'], feature_hashes=hashes,
        limitations='Conditional on verifiable answers. One additional generation seed, not fresh questions '
                    'or independent training seeds. Different exclusions may change the evaluated population.')
    atomic_json(out / 'summary.json', report)
    paired = report['paired']
    lines = [report['scope'], f"Generation seed {report['seed']}; {len(questions)} questions; selected {report['selected_lstd']}",
             f"Retained {report['answers']['retained_answers']}/{report['answers']['total_answers']} answers",
             'method   Brier     accuracy']
    for name, row in results.items():
        lines.append(f"{name:8} {row['test']['brier']:.5f}   {row['test']['accuracy']:.3f}")
    lines += [f"Paired question difference: {paired['question_mean_difference']:+.6f}; 95% interval {paired['question_bootstrap_95pct_interval']}",
              'Negative favors LSTD. Original difference: ' + str(report['original_paired_difference']),
              'See length-report.txt for length/position groups.', report['limitations']]
    (out / 'report.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def initialize(fit, out, seed):
    fit, out = fit.resolve(), out.resolve()
    fitted = json.loads((fit / 'manifest.json').read_text())
    results = json.loads((fit / 'summary.json').read_text())
    source = Path(fitted['source'])
    if out == fit or out == source or fit in out.parents or source in out.parents:
        raise ValueError('Use a separate output directory')
    if seed < 0 or seed > 2**31 - 10000:
        raise ValueError('Invalid generation seed')
    for name, expected in fitted['source_hashes'].items():
        if sha256(source / name) != expected:
            raise ValueError(f'Source changed: {name}')
    original = json.loads((source / 'manifest.json').read_text())
    if (original['config'].get('protocol') != 'frozen-critics-v2'
            or original['config'].get('prefix_sampling') != PREFIX_SAMPLING):
        raise ValueError('Expected the original frozen-critics-v2 random-prefix protocol')
    expected_sampling = dict(temperature=1., top_p=1., top_k=-1, max_tokens=2048, seed='42 + question index')
    if original['config'].get('sampling') != expected_sampling:
        raise ValueError('Original sampling settings differ from the replication protocol')
    original_questions = json.loads((source / 'questions.json').read_text())['questions']
    questions = [dict(q, source_index=i) for i, q in enumerate(original_questions) if q['split'] == 'test']
    if not questions:
        raise ValueError('Missing test questions')
    old_seeds = {original['config'].get('generation_seed', 42) + q['source_index'] for q in questions}
    if old_seeds.intersection(range(seed, seed + len(questions))):
        raise ValueError('New generation seeds overlap the original test run')
    # Require the same model bytes, verifier and generation/extraction libraries.
    for name, expected in original['files'].items():
        if name.startswith('models/qwen-math/') and sha256(ROOT / name) != expected:
            raise ValueError(f'Base model changed: {name}')
    versions = {p: version(p) for p in ('torch', 'transformers', 'vllm', 'datasets', 'numpy')}
    if versions != original['versions']:
        raise ValueError('Use the original pinned container/runtime for seed replication')
    verifier = json.loads(subprocess.check_output(verifier_command() + ['--provenance'], text=True,
                         env=dict(os.environ, PYTHONPATH=str(ROOT / 'src'))))
    if verifier != original['verifier']:
        raise ValueError('Verifier changed; this would not isolate generation-seed variation')
    chosen = torch.load(fit / 'selected-lstd.pt', weights_only=True, map_location='cpu')
    if chosen['method'] != results['selected_lstd']:
        raise ValueError('Selected critic does not match fit summary')
    tracked = [fit / name for name in ('manifest.json', 'summary.json', 'selected-lstd.pt', 'ridge.pt')]
    tracked += [source / name for name in ('manifest.json', 'questions.json', 'normalization.pt')]
    tracked += [ROOT / name for name in original['files'] if name.startswith('models/qwen-math/')]
    tracked += [ROOT / name for name in ('scripts/validate_lstd_seed.py', 'scripts/analyze_lstd.py',
        'scripts/fit_lstd.py', 'scripts/frozen_critics.py', 'src/math_rl/critic_probe.py',
        'src/math_rl/lstd.py', 'src/math_rl/ppo_reward.py', 'src/math_rl/verifier_batch.py',
        'src/math_rl/math_verify_reward.py')]
    manifest = dict(protocol='lstd-seed-validation-v1', fit=str(fit), source=str(source),
        selected_lstd=results['selected_lstd'], versions=versions, verifier=verifier,
        config=dict(responses=original['config']['responses'], generation_seed=seed,
                    sampling=expected_sampling, prefix_sampling=PREFIX_SAMPLING),
        questions=questions, files={str(p): sha256(p) for p in tracked})
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        if json.loads((out / 'manifest.json').read_text()) != manifest:
            raise ValueError('Resume requires unchanged locked inputs, seed, code and runtime')
    else:
        if any(out.iterdir()):
            raise ValueError('Choose an empty output directory')
        atomic_json(out / 'manifest.json', manifest)
        atomic_json(out / 'questions.json', dict(questions=questions))
    for name in ('trajectories', 'features'):
        (out / name).mkdir(exist_ok=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fit', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=314159)
    parser.add_argument('--stage', choices=('generate', 'extract', 'evaluate'))
    args = parser.parse_args()
    out = args.out.resolve()
    if args.stage:
        manifest = json.loads((out / 'manifest.json').read_text())
        # Fail rather than silently using changed weights/code between stages.
        for name, expected in manifest['files'].items():
            if sha256(name) != expected:
                raise ValueError(f'Locked input changed: {name}')
        questions = manifest['questions']
        if args.stage == 'evaluate':
            evaluate(out, manifest, questions)
        else:
            {'generate': generate, 'extract': extract}[args.stage](out, manifest['config'], questions)
        return
    manifest = initialize(args.fit, out, args.seed)
    for stage in ('generate', 'extract', 'evaluate'):
        started = time.monotonic()
        atomic_json(out / 'status.json', dict(stage=stage, state='running'))
        with (out / f'{stage}.log').open('a') as log:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), str(args.fit.resolve()),
                                     '--out', str(out), '--stage', stage], stdout=log, stderr=subprocess.STDOUT)
        atomic_json(out / 'status.json', dict(stage=stage, state='complete' if result.returncode == 0 else 'failed',
                    exit_code=result.returncode, seconds=time.monotonic() - started))
        if result.returncode:
            raise RuntimeError(f'{stage} failed; see {out / (stage + ".log")}')
    print((out / 'report.txt').read_text())


if __name__ == '__main__':
    main()
