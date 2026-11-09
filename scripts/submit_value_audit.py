"""Submit smoke, matched training/pilot, then a separately approved full MC audit.

Stdlib only: run on the login node. Ambiguous sbatch responses stop the chain.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def gate(out, phase):
    record = json.loads((out / phase / 'verification.json').read_text())
    digest = hashlib.sha256((out / 'manifest.json').read_bytes()).hexdigest()
    bank = hashlib.sha256((out / phase / 'bank.json').read_bytes()).hexdigest()
    if not record.get('passed') or record['manifest_sha256'] != digest or record['bank_sha256'] != bank:
        raise ValueError(f'Matching passed {phase} verification is required')
    for name, expected in record.get('files', {}).items():
        if Path(name).name != name or hashlib.sha256((out / phase / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Passed audit artifact changed')


def submit(out, mode, source=None, concurrency=4, updates=120, actor_seed=1001):
    out = Path(out).resolve()
    if mode not in ('smoke', 'pilot', 'full'):
        raise ValueError('Unknown submission mode')
    if not 1 <= concurrency <= 6:
        raise ValueError('Concurrency must be between 1 and 6')
    if type(updates) is not int or updates < 2 or type(actor_seed) is not int or actor_seed < 0:
        raise ValueError('Invalid training updates/actor seed')
    if mode == 'smoke':
        if source is None or not (Path(source) / 'manifest.json').is_file() or out.exists():
            raise ValueError('Smoke requires an existing --source study and a new output directory')
    else:
        gate(out, 'smoke' if mode == 'pilot' else 'pilot')
    ledger = Path(str(out) + '-submission')
    ledger.mkdir(parents=True, exist_ok=True)
    # No automatic submission retry: a scheduler timeout may mean an accepted job.
    (ledger / f'.submission-{mode}').mkdir(exist_ok=False)
    logs = ledger / 'logs'
    logs.mkdir(exist_ok=True)

    def job(stage, dependency='', array=None, gpus=0, hours=1):
        name = f'{mode}-{stage}'
        args = ['sbatch', '--parsable', '--nodes=1', '--ntasks=1', '--cpus-per-task=8', '--mem=128G',
                '--exclude=compsci-cluster-fitz-15', f'--time={hours:02d}:00:00',
                f'--job-name=value-{name}', f'--output={logs}/{name}-%A_%a.out']
        args += ['--partition=compsci-gpu', f'--gres=gpu:a5000:{gpus}'] if gpus else ['--partition=compsci']
        if dependency:
            args += [f'--dependency=afterok:{dependency}', '--kill-on-invalid-dep=yes']
        if array:
            args += [f'--array={array}']
        args += ['scripts/run_value_audit.sh', stage, str(out), mode, str(gpus),
                 str(Path(source).resolve()) if source is not None else '', str(updates), str(actor_seed)]
        sidecar = ledger / f'{name}-submission.json'
        sidecar.write_text(json.dumps(dict(state='submitting', command=args), indent=2) + '\n')
        try:
            result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, timeout=90)
        except (subprocess.TimeoutExpired, OSError) as exc:
            sidecar.write_text(json.dumps(dict(state='ambiguous_or_failed', error=str(exc), command=args), indent=2) + '\n')
            raise RuntimeError(f'{name}: check Slurm before retrying; submission outcome is unknown') from exc
        if result.returncode or not re.fullmatch(r'\d+(;[\w.-]+)?', result.stdout.strip()):
            sidecar.write_text(json.dumps(dict(state='ambiguous_or_failed', stdout=result.stdout,
                                              stderr=result.stderr, command=args), indent=2) + '\n')
            raise RuntimeError(f'{name}: {result.stderr}. Check Slurm before retrying; no automatic resubmission.')
        value = result.stdout.strip().split(';')[0]
        sidecar.write_text(json.dumps(dict(state='submitted', job_id=value, command=args), indent=2) + '\n')
        with (ledger / 'jobs.tsv').open('a') as handle:
            handle.write(f'{name}\t{value}\n')
        print(f'Submitted {name}: {value}', flush=True)
        return value

    dependency = ''
    if mode == 'smoke':
        dependency = job('prepare')
    if mode in ('smoke', 'pilot'):
        dependency = job('train', dependency, f'0-5%{concurrency}', gpus=2,
                         hours=4 if mode == 'smoke' else 12)
        dependency = job('training-gate', dependency)
    dependency = job('bank', dependency, f'0-5%{concurrency}', gpus=1, hours=2)
    dependency = job('pack', dependency)
    if mode == 'full':
        dependency = job('predict', dependency, f'0-5%{concurrency}', gpus=1, hours=2)
        # Separate prediction and reference stages so concurrency remains a global
        # per-stage GPU-task bound, rather than two independent active arrays.
        dependency = job('mc', dependency, f'0-23%{concurrency}', gpus=1, hours=24)
    else:
        dependency = job('evaluate', dependency, f'0-5%{concurrency}', gpus=1,
                         hours=4 if mode == 'smoke' else 12)
    job('report', dependency)
    print(f'Jobs: {ledger}/jobs.tsv\nReport: {out}/{mode}/report.txt\nPlot: {out}/{mode}/value-scatter.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=['smoke', 'pilot', 'full'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--concurrency', type=int, default=4)
    parser.add_argument('--updates', type=int, help='Smoke preparation only; locked thereafter (default 120)')
    parser.add_argument('--actor-seed', type=int, help='Smoke preparation only; locked thereafter (default 1001)')
    args = parser.parse_args()
    if args.mode != 'smoke' and (args.source is not None or args.updates is not None or args.actor_seed is not None):
        parser.error('Set source, updates and actor seed during smoke preparation only; later stages use the manifest')
    submit(args.out, args.mode, args.source, args.concurrency,
           args.updates if args.updates is not None else 120,
           args.actor_seed if args.actor_seed is not None else 1001)


if __name__ == '__main__':
    main()
