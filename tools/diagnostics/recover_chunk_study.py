"""Recover only unmapped-ID failures in a separate, explicitly mixed-code study.

Never modify the original study, its gate, manifest, or partial actor directories.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
ALLOWED_CHANGES = {
    'packages/slice-critic/src/slice_critic/chunks.py',
    'src/math_rl/trainer.py', 'src/math_rl/chunk_study.py',
}


def read(p): return json.loads(p.read_text())
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p, value):
    p.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')


def changed_runtime(old, new):
    changed = sorted(k for k in old.keys() | new.keys() if old.get(k) != new.get(k))
    if not changed or not set(changed) <= ALLOWED_CHANGES:
        raise ValueError(f'Unexpected runtime changes: {changed}; recovery only allows the audited unmapped-ID fix')
    return changed


def retained_files(case, steps, chunk):
    names = ['result.json', 'run.json', 'base-test.jsonl', 'final-test.jsonl',
             'linear-critic/metrics.jsonl', 'linear-critic/statistics.pt', f'linear-critic/{steps}.pt']
    if chunk: names.append('chunk-metrics.jsonl')
    return {name: digest(case/name) for name in names}


def prepare(source, out):
    from math_rl import chunk_study as study
    from math_rl.provenance import snapshot
    from slice_critic.chunks import UnmappedTokenError  # Require the actual fix.
    source = source.resolve(); out = out.absolute()
    if out.exists() or source == out or source in out.parents:
        raise ValueError('Use a new recovery directory outside the original study')
    old = read(source/'manifest.json')
    current = snapshot(ROOT)['files']
    changed = changed_runtime(old['runtime'], current)
    # Verify original assets/gate using the old declared runtime, without editing it.
    study.require_gate(source)
    for name, checksum in old['data_hashes'].items():
        if digest(source/name) != checksum: raise ValueError('Original data changed')
    if digest(source/'filters/sparse.pt') != old['filter_sha256']:
        raise ValueError('Original filter changed')
    records = {}; failed = []
    for task, (arm, seed) in enumerate(study.jobs(old)):
        name = f'{arm}-seed{seed}'; case = source/name
        status = read(source/f'run-{task}-status.json')
        if status['state'] == 'complete':
            study.audit(source, old, arm, seed)
            records[name] = dict(origin='original', task=task, path=str(case),
                files=retained_files(case, old['steps'], old['arms'][arm]['mode']!='token'))
        elif status['state'] == 'failed':
            if (case/'result.json').exists(): raise ValueError('Failed task has a result; inspect before recovery')
            if 'ValueError: Empty or unknown sampled ByteLevel token' not in (source/f'{name}.log').read_text(errors='replace'):
                raise ValueError(f'Task {task} has a different failure')
            failed.append(task)
            records[name] = dict(origin='restarted_with_unmapped_id_fix', task=task,
                original_partial_path=str(case), failure_log_sha256=digest(source/f'{name}.log'))
        else: raise ValueError(f'Task {task} still active or incomplete: {status["state"]}')
    if not failed: raise ValueError('No failed runs to recover')
    prior_smoke = sum(study.audit(source, old, arm, 19, True)['accounted_gpu_hours'] for arm in old['arms'])
    out.mkdir(parents=True, exist_ok=False)
    for folder in ('data','smoke-data','filters'): shutil.copytree(source/folder, out/folder)
    for name, record in records.items():
        if record['origin']=='original': (out/name).symlink_to(record['path'], target_is_directory=True)
    m = dict(old, runtime=current)
    m['recovery'] = dict(source=str(source), source_manifest_sha256=digest(source/'manifest.json'),
        source_runtime=old['runtime'], changed_runtime_files=changed, failed_tasks=failed, runs=records,
        original_smoke_accounted_gpu_hours=prior_smoke, tool_sha256=digest(Path(__file__)),
        disclosure='Mixed-code recovery: completed original runs retained; only software-failed seeds restarted from update zero. No outcome-based replacements. New alignment fallback preserves original action IDs. Original failed allocation costs are additional and not included in per-arm successful-run means.')
    write(out/'manifest.json',m)
    study.validate(out)
    print(f'Preserved {len(records)-len(failed)} completed runs; restart tasks {failed}', flush=True)


def validate(out):
    from math_rl import chunk_study as study
    m = study.validate(out); r = m['recovery']; source = Path(r['source'])
    if digest(Path(__file__)) != r['tool_sha256'] or digest(source/'manifest.json') != r['source_manifest_sha256']:
        raise ValueError('Recovery tool or original manifest changed')
    for name, record in r['runs'].items():
        if record['origin']=='original':
            if (out/name).resolve() != Path(record['path']): raise ValueError('Retained run link changed')
            for file, checksum in record['files'].items():
                if digest(out/name/file) != checksum: raise ValueError(f'Retained artifact changed: {name}/{file}')
        elif digest(source/f'{name}.log') != record['failure_log_sha256']:
            raise ValueError('Original failure log changed')
    return m


def stage(args):
    sys.path[:0] = [str(ROOT/'src'), str(ROOT/'packages/slice-critic/src')]
    from math_rl import chunk_study as study
    out = args.out.absolute()
    if args.stage=='prepare': return prepare(args.source, out)
    m = validate(out); started=time.perf_counter()
    status=out/f'recovery-{args.stage}-{args.task}-status.json'; write(status,dict(state='running'))
    try:
        if args.stage=='smoke': study.execute(out,m,list(m['arms'])[args.task],19,True)
        elif args.stage=='gate': study.gate(out,m)
        elif args.stage=='run':
            if args.task not in m['recovery']['failed_tasks']: raise ValueError('Only failed tasks may restart')
            study.require_gate(out); arm,seed=study.jobs(m)[args.task]; study.execute(out,m,arm,seed)
        elif args.stage=='report':
            study.report(out,m)
            summary=read(out/'summary.json');summary['recovery']=m['recovery']
            summary['additional_original_smoke_accounted_gpu_hours']=m['recovery']['original_smoke_accounted_gpu_hours']
            summary['limitations']+=' '+m['recovery']['disclosure']
            write(out/'summary.json',summary)
            with (out/'report.txt').open('a') as f:
                f.write('\n'+m['recovery']['disclosure']+'\n')
                f.write(f"Additional original smoke accounted GPUh: {m['recovery']['original_smoke_accounted_gpu_hours']:.3f}\n")
                for name, record in m['recovery']['runs'].items():
                    f.write(f'{name}: {record["origin"]}\n')
        write(status,dict(state='complete',seconds=time.perf_counter()-started))
    except BaseException as exc:
        write(status,dict(state='failed',error=str(exc)));raise


def submit(args):
    if not args.source or not args.source.is_dir() or args.out.exists():
        raise ValueError('Require existing --source and a fresh --out')
    if not 1<=args.concurrency<=4: raise ValueError('Recovery concurrency must be 1..4')
    source=args.source.absolute(); out=args.out.absolute(); m=read(source/'manifest.json')
    # Read terminal statuses before any submission; do not guess failed tasks.
    failed=[]
    for task in range(len(m['arms'])*len(m['seeds'])):
        status=read(source/f'run-{task}-status.json')['state']
        if status=='failed': failed.append(task)
        elif status!='complete': raise ValueError(f'Task {task} is not terminal: {status}')
    if not failed: raise ValueError('No failed tasks')
    ledger=Path(str(out)+'-submission');ledger.mkdir(parents=True,exist_ok=True)
    (ledger/'.submission').mkdir(exist_ok=False)
    logs=ledger/'logs';logs.mkdir(exist_ok=True)
    def job(name, dependency=None, array=None, gpu=False):
        cmd=['sbatch','--parsable','--nodes=1','--ntasks=1','--cpus-per-task=8','--mem=128G',
            '--exclude=compsci-cluster-fitz-15',f'--job-name=chunk-recover-{name}',
            f'--time={"12:00:00" if name=="run" else "04:00:00" if gpu else "01:00:00"}',
            f'--output={logs}/{name}-%A_%a.out']
        cmd+=['--partition=compsci-gpu','--gres=gpu:a5000:2'] if gpu else ['--partition=compsci']
        if dependency:cmd += [f'--dependency=afterok:{dependency}','--kill-on-invalid-dep=yes']
        if array:cmd += [f'--array={array}']
        cmd += ['tools/diagnostics/run_chunk_recovery.sh',name,str(out),str(source),'1' if gpu else '0']
        sidecar=ledger/f'{name}-submission.json';write(sidecar,dict(state='submitting',command=cmd))
        result=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True)
        if result.returncode or not re.fullmatch(r'\d+(;[\w.-]+)?',result.stdout.strip()):
            write(sidecar,dict(state='ambiguous_or_failed',stdout=result.stdout,stderr=result.stderr))
            raise RuntimeError('Submission failed/ambiguous: inspect Slurm and sidecar; do not blindly repeat')
        ident=result.stdout.strip().split(';')[0];write(sidecar,dict(state='submitted',job_id=ident,command=cmd))
        with (ledger/'jobs.tsv').open('a') as f:f.write(f'{name}\t{ident}\n')
        print(f'Submitted {name}: {ident}',flush=True);return ident
    prep=job('prepare');smoke=job('smoke',prep,f'0-3%{args.concurrency}',True)
    gate=job('gate',smoke);run=job('run',gate,','.join(map(str,failed))+f'%{min(args.concurrency,len(failed))}',True)
    job('report',run)
    print(f'Recovery report: {out}/report.txt\nJobs: {ledger}/jobs.tsv')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['prepare','smoke','gate','run','report'])
    p.add_argument('--submit',action='store_true');p.add_argument('--out',required=True,type=Path)
    p.add_argument('--source',type=Path);p.add_argument('--task',type=int,default=0)
    p.add_argument('--concurrency',type=int,default=2)
    args=p.parse_args()
    if args.submit==bool(args.stage):p.error('Choose --submit or --stage')
    if (args.submit or args.stage=='prepare') and args.source is None:p.error('--source is required')
    submit(args) if args.submit else stage(args)


if __name__=='__main__':main()
